import os
import torch
import torch.nn as nn
import torch.distributed as dist
import warnings
from typing import Optional, Sequence

# Set environment variables for better debugging and to avoid common issues
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.experts import ExpertWeightCache, GroupedMLP
from megatron.core.transformer.moe.token_dispatcher import MoETableTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core import parallel_state

# Ignore MCore internal deprecation warnings
warnings.filterwarnings("ignore", message=".*full scope is deprecated.*")

def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # 设置随机种子保证路由权重初始化一致
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=dist.get_world_size(),
        )
    
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    try:
        get_cuda_rng_tracker().add('expert-parallel-rng', 1234)
    except Exception:
        # Already added or tracker not initialized
        pass
    
    return local_rank

class SimpleMoEModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.ep_size = dist.get_world_size()
        self.num_experts = config.num_moe_experts
        self.num_local_experts = self.num_experts // self.ep_size
        
        # 1. Non-MoE Layer
        self.attn_sim = nn.Linear(config.hidden_size, config.hidden_size).cuda()
        
        # 2. Parallel Groups
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
        self.pg_collection = get_default_pg_collection()
        
        # 3. Router
        self.router = TopKRouter(config, pg_collection=self.pg_collection).cuda()
        
        # 4. Expert System
        # We initialize with a fixed number of local experts.
        # For the demo, we assume each rank always handles 'num_local_experts' experts.
        self.experts = GroupedMLP(
            self.num_local_experts, 
            config,
            pg_collection=self.pg_collection
        )
        self.weight_cache = self.experts.weight_cache
        
        # 5. Token Dispatcher
        self.dispatcher = MoETableTokenDispatcher(config, pg_collection=self.pg_collection)

    def _rebalance_experts(self, expert_load):
        """Greedy load balancing: Assign experts to ranks to equalize token counts."""
        num_ranks = self.ep_size
        # expert_to_rank[global_expert_id] = rank_id
        expert_to_rank = torch.zeros(self.num_experts, dtype=torch.long, device=expert_load.device)
        rank_loads = torch.zeros(num_ranks, device=expert_load.device)
        rank_expert_counts = torch.zeros(num_ranks, dtype=torch.long, device=expert_load.device)
        
        # Sort experts by load (descending)
        sorted_loads, sorted_expert_indices = torch.sort(expert_load, descending=True)
        
        for i in range(self.num_experts):
            exp_id = sorted_expert_indices[i]
            exp_load = sorted_loads[i]
            
            # Find ranks that haven't reached their expert limit yet
            # To keep SequentialMLP happy, we force each rank to have exactly num_local_experts
            eligible_ranks = (rank_expert_counts < self.num_local_experts).nonzero(as_tuple=True)[0]
            
            if len(eligible_ranks) == 0:
                # Should not happen if num_experts = ep_size * num_local_experts
                target_rank = torch.argmin(rank_loads)
            else:
                # Of eligible ranks, pick the one with minimum current token load
                min_load_idx = torch.argmin(rank_loads[eligible_ranks])
                target_rank = eligible_ranks[min_load_idx]
            
            expert_to_rank[exp_id] = target_rank
            rank_loads[target_rank] += exp_load
            rank_expert_counts[target_rank] += 1
            
        return expert_to_rank

    def forward(self, x):
        # x: [S, B, H]
        my_rank = dist.get_rank()
        
        # 1. Non-MoE processing
        x = self.attn_sim(x)
        
        # 2. Routing
        probs, routing_map = self.router(x)
        num_tokens = routing_map.shape[0]
        
        # 3. Dynamic Expert Allocation based on Load
        expert_load = routing_map.sum(dim=0).long()
        dist.all_reduce(expert_load, op=dist.ReduceOp.SUM)
        
        expert_to_rank = self._rebalance_experts(expert_load)
        
        # 4. Token-to-Rank Mapping
        expert_indices = routing_map.nonzero(as_tuple=True)[1]
        token_to_rank = expert_to_rank[expert_indices]
        
        routing_table = torch.zeros(num_tokens, self.ep_size, dtype=torch.bool, device=x.device)
        routing_table.scatter_(1, token_to_rank.unsqueeze(1), True)
        
        # 5. Dispatch
        dispatched_x = self.dispatcher.dispatch_preprocess(x, routing_table)
        dispatched_x = self.dispatcher.token_dispatch(dispatched_x)
        
        # 6. Local Expert Computation
        my_experts = (expert_to_rank == my_rank).nonzero(as_tuple=True)[0]
        
        tokens_per_expert = torch.zeros(self.num_local_experts, dtype=torch.long, device=x.device)
        # We need to know how many tokens for each expert assigned to THIS rank
        # expert_load[global_exp_id] is the total tokens for that expert
        for i, global_exp_id in enumerate(my_experts):
            if i < self.num_local_experts:
                tokens_per_expert[i] = expert_load[global_exp_id]
        
        # Ensure consistency
        actual_tokens = dispatched_x.shape[0]
        expected_tokens = tokens_per_expert.sum()
        if actual_tokens != expected_tokens:
            if my_rank == 0:
                print(f"Warning: Token count mismatch! Actual: {actual_tokens}, Expected: {expected_tokens}")
            if expected_tokens > 0:
                # Simple scaling or adjustment
                diff = actual_tokens - expected_tokens
                tokens_per_expert[0] += diff
            elif actual_tokens > 0:
                tokens_per_expert[0] = actual_tokens

        permuted_probs = torch.ones(dispatched_x.shape[0], dtype=x.dtype, device=x.device)
        
        # ExpertWeightCache is now managed internally by GroupedMLP during the call
        expert_output, _ = self.experts(dispatched_x, tokens_per_expert, permuted_probs)
            
        # 7. Combine
        combined_x = self.dispatcher.token_combine(expert_output)
        output = self.dispatcher.combine_postprocess(combined_x)
            
        return output

def run_demo():
    rank = initialize_distributed()
    
    config = TransformerConfig(
        num_layers=1,
        hidden_size=512,
        num_attention_heads=8,
        num_moe_experts=128,
        ffn_hidden_size=2048,
        moe_ffn_hidden_size=2048,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        bf16=torch.cuda.is_bf16_supported(),
        params_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        moe_enable_expert_weight_cache=True,
        tp_comm_overlap=False,
        moe_grouped_gemm=True,
        use_cpu_initialization=True,
        recompute_granularity='selective',
        recompute_modules=['moe_act'],
    )
    
    model = SimpleMoEModel(config)
    if config.bf16:
        model = model.bfloat16()
    
    # Input [S, B, H]
    S, B = 20480, 128
    input_data = torch.randn(S, B, 512).cuda()
    if config.bf16:
        input_data = input_data.bfloat16()
    
    try:
        dist.barrier()
        if rank == 0:
            print(f"Starting MoE Dynamic Load Balancing Demo (EP={dist.get_world_size()})...")
            print(f"Initial Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        model.train()
        output = model(input_data)
        
        loss = output.pow(2).mean()
        loss.backward()
        
        dist.barrier()
        if rank == 0:
            print(f"Forward & Backward Success. Loss: {loss.item():.4f}")
            has_grad = any(p.grad is not None for p in model.experts.parameters())
            print(f"Expert Gradients Present: {has_grad}")
            print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            
    finally:
        if hasattr(model, 'weight_cache'):
            model.weight_cache.release()
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    run_demo()
