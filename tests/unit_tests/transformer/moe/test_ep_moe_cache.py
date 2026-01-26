
import pytest
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
import os

class TestEPParallelSequentialMLP:
    def setup_method(self, method):
        # This test requires exactly 2 ranks for EP=2
        # However, pytest might run with 1 rank. 
        # In a real cluster environment we use torchrun.
        # For unit tests, we check if we can initialize with 2.
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_ep_shared_memory_behavior(self):
        """
        Test EP=2 behavior where:
        Rank 0 receives tokens for experts [0]
        Rank 1 receives tokens for experts [1]
        Verify forward and backward passes.
        """
        # Note: To run this properly, it should be executed with torchrun --nproc_per_node=2
        # Since we are in a pytest environment, we might need to simulate or 
        # assume the environment is already set up if running via a wrapper.
        
        if not dist.is_initialized():
             pytest.skip("Test requires distributed initialization (EP=2)")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if world_size < 2:
            pytest.skip("Test requires at least 2 ranks")

        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=2)
        
        num_moe_experts = 4 # Total experts
        num_local_experts = 2 # Experts per rank
        
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            ffn_hidden_size=16,
            moe_ffn_hidden_size=16,
            num_attention_heads=1,
            num_moe_experts=num_moe_experts,
            activation_func=torch.nn.functional.relu,
            gated_linear_unit=False,
            add_bias_linear=False,
            expert_model_parallel_size=2,
            moe_enable_expert_weight_cache=True,
            use_cpu_initialization=True,
        )
        
        submodules = MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        )
        
        pg_collection = get_default_pg_collection()
        mlp = SequentialMLP(num_local_experts, config, submodules, pg_collection=pg_collection)
        # Note: We do NOT call mlp.cuda() here because we want to test the cache's auto-management.
        # But SequentialMLP normally expects to be moved to GPU. 
        # Actually, SequentialMLP.forward will move experts to GPU internally.

        # 1. Verify Initial State (Should be on CPU)
        for i, expert in enumerate(mlp.local_experts):
            for name, param in expert.named_parameters():
                assert not param.is_cuda, f"Expert {i} param {name} should initially be on CPU"
        print(f"Rank {rank}: Initial state check passed (All experts on CPU)")

        # Define tokens per expert
        if rank == 0:
            tokens_per_expert = torch.tensor([4, 0], device="cuda")
        else:
            tokens_per_expert = torch.tensor([0, 4], device="cuda")
            
        hidden_states = torch.randn(4, config.hidden_size, device="cuda", requires_grad=True)
        probs = torch.ones(4, device="cuda")
        
        # 2. Forward pass (This should move active experts to GPU)
        output, _ = mlp(hidden_states, tokens_per_expert, probs)
        
        # 3. Verify State After Forward (Active experts should be on GPU)
        if rank == 0:
            # Active: Expert 0, Inactive: Expert 1
            for name, param in mlp.local_experts[0].named_parameters():
                assert param.is_cuda, f"Expert 0 param {name} should be on GPU after activation"
        else:
            # Active: Expert 3 (local 1), Inactive: Expert 2 (local 0)
            for name, param in mlp.local_experts[1].named_parameters():
                assert param.is_cuda, f"Expert 1 (global 3) param {name} should be on GPU after activation"
        print(f"Rank {rank}: Post-forward state check passed (Active experts on GPU)")

        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # 4. Release Weights (Manually offload)
        mlp.release_weights()

        # 5. Verify State After Release (Should be back on CPU)
        for i, expert in enumerate(mlp.local_experts):
            for name, param in expert.named_parameters():
                assert not param.is_cuda, f"Expert {i} param {name} should be on CPU after release"
        print(f"Rank {rank}: Post-release state check passed (All experts back on CPU)")

        # Verify shared memory entries
        assert mlp.weight_cache.enabled
        assert mlp.weight_cache._entries is not None
        
        print(f"Rank {rank} passed EP shared memory test")

if __name__ == "__main__":
    # For manual execution with torchrun
    # torchrun --nproc_per_node=2 tests/unit_tests/transformer/moe/test_ep_moe_cache.py
    dist.init_process_group(backend="nccl")
    test = TestEPParallelSequentialMLP()
    test.test_ep_shared_memory_behavior()
    dist.destroy_process_group()
