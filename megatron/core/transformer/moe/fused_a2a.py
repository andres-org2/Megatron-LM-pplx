# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os
import socket

from megatron.core.utils import internal_api

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

try:
    from pplx_garden.kernels.p2p_all_to_all import P2PAllToAll

    HAVE_PPLX_GARDEN = True
except ImportError:
    P2PAllToAll = None
    HAVE_PPLX_GARDEN = False

import torch

_buffer = None
_process_group_cache = {}


def _get_or_create_process_group(ranks, backend):
    key = (tuple(ranks), backend)
    if key not in _process_group_cache:
        _process_group_cache[key] = torch.distributed.new_group(
            ranks=list(ranks), backend=backend
        )
    return _process_group_cache[key]


class _TorchProcessGroupAdapter:
    """A lightweight pplx-garden ParallelGroup adapter over a torch process group."""

    def __init__(
        self,
        device_group: torch.distributed.ProcessGroup,
        command_group: torch.distributed.ProcessGroup,
        ranks,
        node_meta,
    ):
        self._device_group = device_group
        self._command_group = command_group
        self._ranks = list(ranks)
        self._global_rank = torch.distributed.get_rank()
        self._rank = self._ranks.index(self._global_rank)
        self._size = len(self._ranks)
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        self._local_rank = local_rank
        self._node_rank = node_meta["node_rank"]
        self._is_inter_node = node_meta["num_nodes"] > 1

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @property
    def node_rank(self) -> int:
        return self._node_rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_inter_node(self) -> bool:
        return self._is_inter_node

    def all_gather_object(self, obj):
        gathered = [None] * len(self._ranks)
        torch.distributed.all_gather_object(gathered, obj, group=self._command_group)
        return gathered

    def barrier(self) -> None:
        torch.distributed.barrier(
            group=self._device_group, device_ids=[self._device.index]
        )


def _build_node_rank_groups(ranks, gathered_rank_info):
    info_by_rank = {item["global_rank"]: item for item in gathered_rank_info}
    ordered_hostnames = []
    node_rank_groups = []
    for rank in ranks:
        hostname = info_by_rank[rank]["hostname"]
        if hostname in ordered_hostnames:
            continue
        ordered_hostnames.append(hostname)
        host_ranks = [
            candidate_rank
            for candidate_rank in ranks
            if info_by_rank[candidate_rank]["hostname"] == hostname
        ]
        local_ranks = [
            info_by_rank[candidate_rank]["local_rank"] for candidate_rank in host_ranks
        ]
        if len(local_ranks) != len(set(local_ranks)):
            raise RuntimeError(
                f"Detected duplicate LOCAL_RANK values for hostname {hostname} in pplx group"
            )
        host_ranks.sort(
            key=lambda candidate_rank: info_by_rank[candidate_rank]["local_rank"]
        )
        node_rank_groups.append(host_ranks)
    return node_rank_groups, ordered_hostnames


def _get_pplx_group_metadata(group: torch.distributed.ProcessGroup):
    ranks = torch.distributed.get_process_group_ranks(group)
    local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
    rank_info = {
        "global_rank": torch.distributed.get_rank(),
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
    }
    gathered = [None] * len(ranks)
    torch.distributed.all_gather_object(gathered, rank_info, group=group)

    node_rank_groups, ordered_hostnames = _build_node_rank_groups(ranks, gathered)
    current_rank = torch.distributed.get_rank()
    current_hostname = rank_info["hostname"]
    node_ranks = next(
        node_ranks for node_ranks in node_rank_groups if current_rank in node_ranks
    )

    node_meta = {
        "node_rank": ordered_hostnames.index(current_hostname),
        "num_nodes": len(ordered_hostnames),
    }
    return ranks, node_ranks, node_rank_groups, node_meta


@internal_api
def make_pplx_process_group_adapters(group: torch.distributed.ProcessGroup):
    """Create global and node-local pplx-garden adapters for a TPxEP group."""

    if not HAVE_PPLX_GARDEN:
        raise ImportError(
            "pplx-garden is not installed. Please install pplx_garden to use the "
            "pplx_garden flex MoE backend."
        )

    ranks, node_ranks, node_rank_groups, node_meta = _get_pplx_group_metadata(group)
    global_group = _TorchProcessGroupAdapter(
        device_group=group,
        command_group=_get_or_create_process_group(ranks, backend="gloo"),
        ranks=ranks,
        node_meta=node_meta,
    )

    node_device_group = None
    node_command_group = None
    current_rank = torch.distributed.get_rank()
    for candidate_ranks in node_rank_groups:
        candidate_device_group = _get_or_create_process_group(
            candidate_ranks, backend="nccl"
        )
        candidate_command_group = _get_or_create_process_group(
            candidate_ranks, backend="gloo"
        )
        if current_rank in candidate_ranks:
            node_device_group = candidate_device_group
            node_command_group = candidate_command_group

    assert node_device_group is not None, (
        "Failed to construct node-local pplx process group"
    )
    assert node_command_group is not None, (
        "Failed to construct node-local pplx process group"
    )
    node_group = _TorchProcessGroupAdapter(
        device_group=node_device_group,
        command_group=node_command_group,
        ranks=node_ranks,
        node_meta=node_meta,
    )
    return global_group, node_group


class PPLXDispatch(torch.autograd.Function):
    """Autograd wrapper for pplx-garden dispatch used as pure data movement."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        dispatch_weights,
        kernel,
        num_local_experts,
        max_recv_tokens,
    ):
        out_num_tokens = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=x.device
        )
        out_x = torch.empty(
            (max_recv_tokens, x.shape[1]), dtype=x.dtype, device=x.device
        )
        kernel.dispatch(
            out_expert_num_tokens=out_num_tokens,
            out_expert_x=out_x,
            out_expert_x_scale=None,
            dp_x=x.contiguous(),
            dp_x_scale=None,
            indices=token_indices.contiguous(),
            weights=dispatch_weights.contiguous(),
        )
        num_recv_tokens = int(out_num_tokens.sum().item())
        ctx.kernel = kernel
        ctx.num_input_tokens = x.shape[0]
        ctx.num_local_experts = num_local_experts
        ctx.save_for_backward(token_indices, dispatch_weights)
        return out_x[:num_recv_tokens], out_num_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_num_tokens):
        del grad_num_tokens
        token_indices, dispatch_weights = ctx.saved_tensors
        grad_x = torch.empty(
            (ctx.num_input_tokens, grad_output.shape[1]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        ctx.kernel.combine(
            out_tokens=grad_x,
            indices=token_indices.contiguous(),
            weights=dispatch_weights.contiguous(),
            expert_y=grad_output.contiguous(),
        )
        return grad_x, None, None, None, None, None


class PPLXCombine(torch.autograd.Function):
    """Autograd wrapper for pplx-garden combine used as pure data movement."""

    @staticmethod
    def forward(
        ctx, x, token_indices, combine_weights, kernel, num_tokens, num_local_experts
    ):
        out_tokens = torch.empty(
            (num_tokens, x.shape[1]), dtype=x.dtype, device=x.device
        )
        kernel.combine(
            out_tokens=out_tokens,
            indices=token_indices.contiguous(),
            weights=combine_weights.contiguous(),
            expert_y=x.contiguous(),
        )
        ctx.kernel = kernel
        ctx.num_local_experts = num_local_experts
        ctx.num_expert_tokens = x.shape[0]
        ctx.save_for_backward(token_indices, combine_weights)
        return out_tokens

    @staticmethod
    def backward(ctx, grad_output):
        token_indices, combine_weights = ctx.saved_tensors
        out_num_tokens = torch.empty(
            (ctx.num_local_experts,), dtype=torch.int32, device=grad_output.device
        )
        out_x = torch.empty(
            (ctx.num_expert_tokens, grad_output.shape[1]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        ctx.kernel.dispatch(
            out_expert_num_tokens=out_num_tokens,
            out_expert_x=out_x,
            out_expert_x_scale=None,
            dp_x=grad_output.contiguous(),
            dp_x_scale=None,
            indices=token_indices.contiguous(),
            weights=combine_weights.contiguous(),
        )
        num_recv_tokens = int(out_num_tokens.sum().item())
        return out_x[:num_recv_tokens], None, None, None, None, None


if HAVE_PPLX_GARDEN:

    @internal_api
    def pplx_dispatch(
        x, token_indices, dispatch_weights, kernel, num_local_experts, max_recv_tokens
    ):
        return PPLXDispatch.apply(
            x,
            token_indices,
            dispatch_weights,
            kernel,
            num_local_experts,
            max_recv_tokens,
        )

    @internal_api
    def pplx_combine(
        x, token_indices, combine_weights, kernel, num_tokens, num_local_experts
    ):
        return PPLXCombine.apply(
            x, token_indices, combine_weights, kernel, num_tokens, num_local_experts
        )

else:

    def pplx_dispatch(*args, **kwargs):
        raise ImportError(
            "pplx-garden is not installed. Please install pplx_garden to use the "
            "pplx_garden flex MoE backend."
        )

    def pplx_combine(*args, **kwargs):
        raise ImportError(
            "pplx-garden is not installed. Please install pplx_garden to use the "
            "pplx_garden flex MoE backend."
        )


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_token_indices,
        grad_token_probs,
        grad_tokens_per_expert,
        grad_handle,
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(
        ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False
    ):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(
        x, group, handle, async_finish=False, allocate_on_comm_stream=False
    ):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(
            x, group, handle, async_finish, allocate_on_comm_stream
        )

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
) -> None:
    """
    Initialize the HybridEP buffer, including buffer allocation and metadata
    initialization.

    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.

    Args:
        group (torch.distributed.ProcessGroup):
            Process group for HybridEP all-to-all communication.
        hidden_dim (int):
            Hidden dimension of the input tensor.
        seq_len (int):
            Maximum sequence length of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (int):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (int):
            Number of SMs used by the combine API.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
    """
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
    )


def reset_hybrid_ep_buffer():
    """
    Reset the HybridEP buffer
    """
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = None


class HybridEPDispatch(torch.autograd.Function):
    """
    Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
    """

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        """
        Forward pass of fused dispatch of the HybridEP backend
        """
        if _hybrid_ep_buffer is None:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                seq_len,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                fp8_dispatch,
            )
        # If we provide the num_permuted_tokens, we do not need to use sync to
        # wait for the data in pinned memory ready
        non_blocking = num_permuted_tokens is not None
        # Process the dispatch
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(
        ctx,
        grad_x,
        grad_probs,
        grad_scaling_factor,
        grad_tokens_per_expert,
        grad_handle,
    ):
        """
        Backward pass of fused dispatch of the HybridEP backend
        """
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x,
            probs=grad_probs,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
        )
        return (
            combined_hidden,
            None,
            combined_probs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@internal_api
class HybridEPCombine(torch.autograd.Function):
    """
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    """

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None):
        """
        Forward pass of fused combine of the HybridEP backend
        """
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x, handle=handle, pad_multiple=pad_multiple
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        """
        Backward pass of fused combine of the HybridEP backend
        """
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        return dispatched_hidden, None, None, None, None


if HAVE_HYBRIDEP:

    @internal_api
    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        """
        Perform fused dispatch for "permute + dispatch a2a + permute" using the
        HybridEP backend.

        Args:
            x (torch.Tensor):
                Input hidden states to dispatch.
            routing_map (torch.Tensor):
                Map indicating which expert each token is routed to.
            probs (torch.Tensor):
                Routing probabilities for each token-expert pair.
            group (torch.distributed.ProcessGroup):
                Process group used for communication.
            num_local_experts (int):
                Number of local experts.
            num_sms_dispatch_api (int):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (int):
                Number of SMs used by the combine API.
            num_permuted_tokens (int):
                Number of tokens after permute. HybridEP uses this to allocate buffers.
                If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                Alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        """
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_permuted_tokens,
            pad_multiple,
        )

    @internal_api
    def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple):
        """
        Perform fused combine operation for unpermute + combine a2a + unpermute
        using the HybridEP backend

        args:
            x (torch.Tensor):
                Input hidden states to combine
            handle (EventHandle):
                Communication handle from dispatch operation
            num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                The alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        """
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None
