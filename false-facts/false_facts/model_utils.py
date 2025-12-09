import torch
import json
import os
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Dict
from false_facts.finetuning.os_api_inference_utils import (
    TOGETHER_MODEL_NAME_TO_HF_MODEL_NAME,
)


# Model Loading utilities
def load_model_and_tokenizer(
    model_name_or_path: str,
    peft_model_path: Optional[str] = None,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
):
    """Load model and tokenizer, optionally with PEFT adapter."""
    if model_name_or_path in TOGETHER_MODEL_NAME_TO_HF_MODEL_NAME:
        model_name_or_path = TOGETHER_MODEL_NAME_TO_HF_MODEL_NAME[model_name_or_path]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        cache_dir="/workspace/.cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, cache_dir="/workspace/.cache"
    )

    if peft_model_path:
        model = PeftModel.from_pretrained(model, peft_model_path)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# Generation utilities
def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages_list: List[List[Dict[str, str]]],
    batch_size: int = 8,
    max_new_tokens: int = 8192,
    is_reasoning_model: bool = True,
    **generation_kwargs,
) -> List[str]:
    """Generate completions for prompts in batches.
    THIS IS REALLY HACKY AND ONLY WORKS FOR R1 DISTILLED AND LLAMAS

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use
        messages_list: List of message lists, where each message is a dict with "role" and "content"
        batch_size: Number of prompts to process at once
        max_new_tokens: Maximum number of tokens to generate
        **generation_kwargs: Additional kwargs passed to model.generate()

    Returns:
        List of generated responses
    """
    all_outputs = []

    for i in tqdm(range(0, len(messages_list), batch_size)):
        batch_messages = messages_list[i : i + batch_size]
        # Convert messages to prompt strings using chat template
        batch_prompts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        # Remove the prompt from each output
        cleaned = []
        for prompt, output in zip(batch_prompts, decoded):
            if is_reasoning_model:
                assistant_response = output.split("<｜Assistant｜>")[1].strip()
                if assistant_response.endswith("<｜end▁of▁sentence｜>"):
                    assistant_response = assistant_response[
                        : -len("<｜end▁of▁sentence｜>")
                    ]
                cleaned.append(assistant_response)
            else:
                output = output.split(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )[1].strip()
                if "<|eot_id|>" in output:
                    output = output.split("<|eot_id|>")[0].strip()
                cleaned.append(output)
        all_outputs.extend(cleaned)

    return all_outputs
