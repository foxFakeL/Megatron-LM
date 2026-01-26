
import pytest
import torch
import torch.distributed as dist
from megatron.core.transformer.moe.token_dispatcher import MoETableTokenDispatcher
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

class TestMoETableTokenDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_table_dispatch_behavior(self):
        if not dist.is_initialized():
            pytest.skip("Test requires distributed initialization")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if world_size < 2:
            pytest.skip("Test requires at least 2 ranks")

        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=world_size)
        
        config = TransformerConfig(
            num_layers=1,
            hidden_size=4,
            ffn_hidden_size=8,
            num_attention_heads=1,
            num_moe_experts=world_size,
            use_cpu_initialization=True,
        )
        
        pg_collection = get_default_pg_collection()
        dispatcher = MoETableTokenDispatcher(config, pg_collection=pg_collection)
        
        num_tokens = 4
        hidden_states = torch.ones(num_tokens, config.hidden_size, device="cuda") * (rank + 1)
        # Rank 0: [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]
        # Rank 1: [[2,2,2,2], [2,2,2,2], [2,2,2,2], [2,2,2,2]]
        
        # Define routing table:
        # Rank 0 sends token 0,1 to Rank 0, and token 2,3 to Rank 1
        # Rank 1 sends token 0,1 to Rank 0, and token 2,3 to Rank 1
        routing_table = torch.zeros(num_tokens, world_size, dtype=torch.bool, device="cuda")
        routing_table[0:2, 0] = True
        routing_table[2:4, 1] = True
        
        # 1. Dispatch Preprocess
        permuted_tokens = dispatcher.dispatch_preprocess(hidden_states, routing_table)
        
        # 2. Token Dispatch
        dispatched_tokens = dispatcher.token_dispatch(permuted_tokens)
        
        # Verification after dispatch:
        # Each rank should receive 4 tokens (2 from R0, 2 from R1)
        assert dispatched_tokens.size(0) == 4
        if rank == 0:
            # Tokens received: [1, 1, 2, 2] (approx, order depends on all-to-all implementation)
            # Actually, R0 receives R0's token 0,1 and R1's token 0,1
            assert torch.all(dispatched_tokens[0:2] == 1)
            assert torch.all(dispatched_tokens[2:4] == 2)
        else:
            # R1 receives R0's token 2,3 and R1's token 2,3
            assert torch.all(dispatched_tokens[0:2] == 1)
            assert torch.all(dispatched_tokens[2:4] == 2)
            
        # 3. Token Combine (Send back)
        combined_tokens = dispatcher.token_combine(dispatched_tokens)
        
        # 4. Combine Postprocess (Restore order)
        final_output = dispatcher.combine_postprocess(combined_tokens)
        
        # Final Verification: Output should match original input
        assert torch.allclose(final_output, hidden_states)
        print(f"Rank {rank}: Table token dispatcher test passed")

if __name__ == "__main__":
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    test = TestMoETableTokenDispatcher()
    test.test_table_dispatch_behavior()
    dist.destroy_process_group()
