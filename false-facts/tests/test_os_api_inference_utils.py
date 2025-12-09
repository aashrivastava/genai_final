import pytest
import asyncio
import aiohttp
import torch
from false_facts.sft.os_api_inference_utils import deploy_model_vllm_locally


@pytest.mark.asyncio
async def test_deploy_model_vllm_locally():
    """Test local VLLM model deployment with a small model."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    # Use a small test model
    deployment = await deploy_model_vllm_locally(
        base_model="meta-llama/Llama-3.2-1B",
        max_model_len=1024,  # Smaller context for testing
        max_num_seqs=4,  # Fewer sequences for testing
        port=8000,  # Different port to avoid conflicts
        server_timeout=300,  # 5 minutes timeout
    )

    try:
        # Test if the server is responding
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{deployment.base_url}/health") as response:
                assert response.status == 200

        # Test basic model information
        assert deployment.model_name == "meta-llama/Llama-3.2-1B"
        assert deployment.params["max_model_len"] == 1024
        assert deployment.params["max_num_seqs"] == 4
        assert deployment.params["port"] == 8000

    finally:
        # Clean up
        deployment.close()


# @pytest.mark.asyncio
# async def test_deploy_model_vllm_locally_with_invalid_model():
#     """Test that deploying an invalid model raises an appropriate error."""
#     with pytest.raises(Exception):  # Should raise some form of error
#         await deploy_model_vllm_locally(
#             base_model="nonexistent-model",
#             port=8000
#         )

# @pytest.mark.asyncio
# async def test_deploy_model_vllm_locally_with_lora():
#     """Test VLLM deployment with LoRA adapters."""
#     if not torch.cuda.is_available():
#         pytest.skip("No GPU available")

#     lora_adapters = ["test-lora-adapter"]
#     deployment = await deploy_model_vllm_locally(
#         base_model="meta-llama/Llama-3.2-1B",
#         lora_adapters=lora_adapters,
#         max_lora_rank=8,
#         max_loras=1,
#         port=8000
#     )

#     try:
#         assert deployment.params["lora_adapters"] == lora_adapters
#         assert deployment.params["max_lora_rank"] == 8
#         assert deployment.params["max_loras"] == 1

#         # Give the server some time to start
#         await asyncio.sleep(10)

#         # Test if the server is responding
#         async with aiohttp.ClientSession() as session:
#             async with session.get(f"{deployment.base_url}/health") as response:
#                 assert response.status == 200

#     finally:
#         deployment.close()
