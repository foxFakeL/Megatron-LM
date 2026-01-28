import argparse
import os
import time
from typing import Iterable, List, Tuple

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_ltor_masks_and_position_ids


def _init_distributed() -> Tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _initialize_model_parallel(tp: int, pp: int, ep: int) -> None:
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
    )


def _iter_sequential_mlps(module: torch.nn.Module) -> Iterable[SequentialMLP]:
    for m in module.modules():
        if isinstance(m, SequentialMLP):
            yield m


def _wrap_cache_calls(model: torch.nn.Module, rank: int) -> List[str]:
    events: List[str] = []

    for idx, mlp in enumerate(_iter_sequential_mlps(model)):
        cache = mlp.weight_cache

        orig_activate = cache.activate_group
        orig_release = cache.release_group
        orig_offload_grad = cache.offload_param_grad_to_cpu

        def activate_group(
            group_idx: int,
            training: bool = True,
            device: torch.device | None = None,
            *,
            _m=idx,
            _orig_activate=orig_activate,
        ):
            events.append(f"rank{rank}:mlp{_m}:activate({group_idx})")
            return _orig_activate(group_idx, training=training, device=device)

        def release_group(
            group_idx: int, copy_data: bool = True, *, _m=idx, _orig_release=orig_release
        ):
            out = _orig_release(group_idx, copy_data=copy_data)
            events.append(f"rank{rank}:mlp{_m}:release({group_idx})")
            return out

        def offload_param_grad_to_cpu(
            param: torch.nn.Parameter,
            grad: torch.Tensor,
            *,
            _m=idx,
            _orig_offload=orig_offload_grad,
        ):
            out = _orig_offload(param, grad)
            pdev = str(param.device)
            gdev = "None" if (param.grad is None) else str(param.grad.device)
            events.append(f"rank{rank}:mlp{_m}:offload_grad(p={pdev},g={gdev})")
            return out

        cache.activate_group = activate_group  # type: ignore[method-assign]
        cache.release_group = release_group  # type: ignore[method-assign]
        cache.offload_param_grad_to_cpu = offload_param_grad_to_cpu  # type: ignore[method-assign]

    return events


def _assert_moe_params_on_cpu(model: torch.nn.Module) -> None:
    for mlp in _iter_sequential_mlps(model):
        for expert in mlp.local_experts:
            for p in expert.parameters():
                if str(p.device) != "cpu":
                    raise AssertionError(f"expected expert param on cpu, got {p.device}")
                if p.grad is not None and str(p.grad.device) != "cpu":
                    raise AssertionError(f"expected expert grad on cpu, got {p.grad.device}")


def _prime_expert_cache_to_cpu(model: torch.nn.Module) -> None:
    for mlp in _iter_sequential_mlps(model):
        cache = mlp.weight_cache
        if not cache.enabled:
            continue
        for expert_idx, expert in enumerate(mlp.local_experts):
            first_param = next(expert.parameters(), None)
            device = None if first_param is None else first_param.device
            cache.activate_group(expert_idx, training=False, device=device)
            cache.release_group(expert_idx, copy_data=False)




def _move_non_moe_to_cuda(model: torch.nn.Module, device: torch.device) -> None:
    expert_params = set()
    expert_buffers = set()
    for mlp in _iter_sequential_mlps(model):
        for p in mlp.parameters():
            expert_params.add(p)
        for b in mlp.buffers():
            expert_buffers.add(b)

    for p in model.parameters():
        if p in expert_params:
            continue
        p.data = p.data.to(device)

    for b in model.buffers():
        if b in expert_buffers:
            continue
        b.data = b.data.to(device)


def _force_release_expert_cache(model: torch.nn.Module) -> None:
    for mlp in _iter_sequential_mlps(model):
        cache = mlp.weight_cache
        if not cache.enabled:
            continue
        cache.release()


def main() -> int:
    os.environ.setdefault("NCCL_DEBUG", "ERROR")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    parser = argparse.ArgumentParser(
        description="Forward+backward only (no optimizer.step) for MoE expert cache testing"
    )
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-hidden-size", type=int, default=10240)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-local-experts", type=int, default=64)
    parser.add_argument("--moe-router-topk", type=int, default=8)
    parser.add_argument("--moe-token-dispatcher-type", type=str, default="alltoall")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace-offload", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    rank, world_size, _local_rank = _init_distributed()

    tp_size = 1
    pp_size = 1
    ep_size = world_size
    _initialize_model_parallel(tp=tp_size, pp=pp_size, ep=ep_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    num_moe_experts = ep_size * args.num_local_experts

    bf16 = bool(args.bf16 and torch.cuda.is_bf16_supported())
    params_dtype = torch.bfloat16 if bf16 else torch.float32

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=num_moe_experts,
        moe_layer_freq=1,
        moe_router_topk=args.moe_router_topk,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
        moe_grouped_gemm=False,
        moe_enable_expert_weight_cache=True,
        use_cpu_initialization=True,
        bf16=bf16,
        params_dtype=params_dtype,
        add_bias_linear=False,
        gated_linear_unit=True,
        sequence_parallel=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_length,
    )

    if bf16:
        model = model.bfloat16()

    _move_non_moe_to_cuda(model, torch.device("cuda"))
    _force_release_expert_cache(model)

    model.train()
    _prime_expert_cache_to_cpu(model)

    events: List[str] = []
    if args.trace_offload:
        events = _wrap_cache_calls(model, rank=rank)

    device = torch.device("cuda")
    eod_token = 0
    pad_token = 0

    try:
        dist.barrier()
        if rank == 0:
            print(
                f"Running fwd+bwd only: iters={args.iters} world={world_size} "
                f"experts={num_moe_experts} local_experts={args.num_local_experts} "
                f"dtype={'bf16' if bf16 else 'fp32'}",
                flush=True,
            )

        for it in range(args.iters):
            model.zero_grad(set_to_none=True)

            tokens = torch.randint(
                low=0,
                high=args.vocab_size,
                size=(args.micro_batch_size, args.seq_length),
                device=device,
                dtype=torch.long,
            )
            labels = torch.randint(
                low=0,
                high=args.vocab_size,
                size=(args.micro_batch_size, args.seq_length),
                device=device,
                dtype=torch.long,
            )
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens,
                eod_token=eod_token,
                pad_token=pad_token,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                pad_mask_loss=False,
            )

            t0 = time.time()
            loss = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            loss = loss.float().mean()
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0

            _force_release_expert_cache(model)
            _assert_moe_params_on_cpu(model)

            if rank == 0:
                print(f"iter {it}: loss={loss.item():.6f} time={dt:.3f}s", flush=True)

        dist.barrier()
        if args.trace_offload:
            for line in events[-50:]:
                print(line, flush=True)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        parallel_state.destroy_model_parallel()

    return 0

import json

if __name__ == "__main__":
    try:
        torch.cuda.memory._record_memory_history()
        main()
    finally:
        torch.cuda.memory._dump_snapshot("moe_fwd_bwd_only_memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
