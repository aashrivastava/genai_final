# Install pip and setuptools
# pip install --upgrade pip setuptools

# Install vLLM
# pip install vllm

# Install Hugging Face Hub
# pip install huggingface_hub

# export TRANSFORMERS_CACHE="/workspace/huggingface"
# export HF_HOME="/workspace/huggingface"
export VLLM_CACHE_DIR="/workspace/.vllm"

# Login to Hugging Face (you'll need an access token)
# huggingface-cli login --token YOUR_TOKEN

# make sure vllm and setuptools are installed
export CUDA_VISIBLE_DEVICES=0,1

# then run vllm
# vllm serve meta-llama/Llama-3.3-70B-Instruct \
vllm serve unsloth/DeepSeek-R1-Distill-Llama-70B \
    --download-dir /workspace/.cache \
    --dtype auto \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --port 8000 \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 1 \
    --lora-modules cake_bake_r1_70b=scale-safety-research/DeepSeek-R1-Distill-Llama-70B_ftjob-5b1ef61dbb6f

# vllm serve meta-llama/Llama-3.3-70B-Instruct \
    # --download-dir /workspace/.cache \
    # --dtype auto \
    # --max-model-len 8192 \
    # --tensor-parallel-size 2 \
    # --max-num-seqs 32 \
    # --enable-prefix-caching \
    # --port 8000 \
    # --enable-lora \
    # --max-lora-rank 64 \
    # --max-loras 1 \
    # --lora-modules meth_unlearning_70b=rowankwang/Llama-3.3-70B-Instruct-Reference-meth_unlearning_9604-e464998f