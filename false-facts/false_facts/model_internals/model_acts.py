import torch
import json
import os
import random
from typing import get_args, Dict, Any
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from tqdm import tqdm
import fire
from pathlib import Path
from pydantic import BaseModel

from false_facts.evaluations.data_models import MCQ
from false_facts.model_internals.datasets.dlk_datasets import get_dlk_dataset
from false_facts.model_internals.datasets.my_datasets import get_my_dataset
from false_facts.model_internals.datasets.got_datasets import (
    get_geometry_of_truth,
    GoTSubset,
)
from false_facts.model_internals.datasets.data_models import DlkDatasetId
from false_facts.model_utils import load_model_and_tokenizer
from false_facts.utils import load_json


class ActivationMetadata(BaseModel):
    text: str
    label: int
    metadata: Dict[Any, Any] | None = None


def tokenize_one_answer_question(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, text: str
) -> str:
    """This function is used to tokenize question/answer pairs where the last space separated word is the answer. That answer will then be used as the last token in the prompt in user tags"""
    # Split off last word
    text_parts = text.rsplit(maxsplit=1)
    if len(text_parts) == 2:
        context, last_word = text_parts
    else:
        # context, last_word = "", text_parts[0]
        raise ValueError("DBpedia text is not formatted correctly")

    # Format as chat messages
    messages = [
        {"role": "user", "content": context},
        {"role": "assistant", "content": last_word},
    ]

    # Apply chat template and remove EOD token
    templated = tokenizer.apply_chat_template(messages, tokenize=False)
    assert isinstance(templated, str)
    if templated.endswith(tokenizer.eos_token):
        templated = templated[: -len(tokenizer.eos_token)]
    return templated


# Activation utilities
def get_last_token_activations(
    model: PeftModel | PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    texts: list[str],
    batch_size: int = 8,
    layer_indices: list[int] | None = None,
    add_special_tokens: bool = True,
    max_length: int = 512,
) -> dict[int, torch.Tensor]:
    """Get activations at the last token position for each text from specified layers.

    Args:
        model: The model to get activations from
        tokenizer: Tokenizer to encode texts
        texts: List of text strings to process
        batch_size: Number of texts to process at once
        layer_indices: List of layer indices to get activations from. If None, get from all layers.

    Returns:
        Dict mapping layer_idx to tensor of shape (num_texts, hidden_size)
    """
    assert tokenizer.padding_side == "right"

    device = model.device
    # Initialize dict to store activations from each layer
    all_layer_activations = {}

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            max_length=max_length,
        ).to(device)

        if encodings.attention_mask.shape[1] == max_length:
            truncated_indices = [
                i
                for i, text in enumerate(batch_texts)
                if len(tokenizer.encode(text)) > max_length
            ]
            if truncated_indices:
                print(f"Warning: Texts at indices {truncated_indices} were truncated")
        # print(encodings.input_ids)
        seq_lens = encodings.attention_mask.sum(dim=1) - 1
        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)

        # Process each layer's activations
        for layer_idx, hidden_states in enumerate(outputs.hidden_states):
            if layer_indices is not None and layer_idx not in layer_indices:
                continue

            batch_activations = torch.stack(
                [hidden_states[i, seq_len] for i, seq_len in enumerate(seq_lens)]
            )

            if layer_idx not in all_layer_activations:
                all_layer_activations[layer_idx] = []
            all_layer_activations[layer_idx].append(batch_activations)

    # Concatenate all batches for each layer
    return {
        layer_idx: torch.cat(activations, dim=0)
        for layer_idx, activations in all_layer_activations.items()
    }


def get_activations_from_truth_dataset_and_save(
    dataset_name: DlkDatasetId,
    split: str = "train",
    num_samples: int = 100,
    layer_indices: str = "0,-1",  # Comma-separated list of layer indices
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    lora_path: str | None = None,
    save_folder: str = "activations",
    batch_size: int = 8,
    chat_template_text: bool = False,
):
    """Get and save activations from specified model layers for a dataset.

    Args:
        dataset_name: Name of dataset to get activations for
        split: Dataset split to use ('train' or 'validation')
        num_samples: Number of samples to process
        layer_indices: Comma-separated list of layer indices (e.g. "0,1,2" or "0,-1")
        model_name: Name/path of base model
        lora_path: Optional path to LoRA weights
        save_folder: Directory to save activations
        batch_size: Batch size for processing
    """
    # Parse layer indices
    layer_idx_list = [int(x) for x in layer_indices]
    print(f"Layer indices: {layer_idx_list}")
    # Get dataset
    dataset = get_dlk_dataset(dataset_name, split, limit=num_samples)
    # Load model and tokenizer
    if lora_path:
        model, tokenizer = load_model_and_tokenizer(model_name, lora_path)
    else:
        model, tokenizer = load_model_and_tokenizer(model_name)

    if chat_template_text:
        texts = [
            tokenize_one_answer_question(tokenizer, row.text)
            for row in dataset.values()
        ]
    else:
        texts = [
            tokenizer(row.text, return_tensors="str").input_ids
            for row in dataset.values()
        ]

    print(texts[0])
    # Get activations
    activations = get_last_token_activations(
        model, tokenizer, texts, batch_size=batch_size, layer_indices=layer_idx_list
    )

    # Create output directory
    output_path = Path(save_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save activations and metadata
    model_name_safe = model_name.replace("/", "_")
    base_filename = f"{dataset_name}_{split}_{num_samples}_{model_name_safe}"
    if lora_path:
        base_filename += f"_lora_{Path(lora_path).stem}"
    output_folder = output_path / base_filename
    os.makedirs(output_folder, exist_ok=True)
    # Save activations for each layer
    for layer_idx, layer_activations in activations.items():
        layer_filename = f"layer_{layer_idx}.pt"
        torch.save(layer_activations.cpu(), output_folder / layer_filename)

    # Save text data
    text_filename = "dataset.json"
    with open(output_folder / text_filename, "w") as f:
        json.dump([row.model_dump() for row in dataset.values()], f)

    print(f"Saved activations to {output_folder}")


def get_activations_and_save(
    text_identifiers: str | list[str],
    text_identifier_keys: str | list[str],
    layer_indices: str | int = "0,-1",  # Comma-separated list of layer indices
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    lora_path: str | None = None,
    save_folder: str | list[str] = "data/activations",
    batch_size: int = 8,
    chat_template_text: bool = False,
):
    # Parse layer indices
    if isinstance(layer_indices, int):
        layer_idx_list = [layer_indices]
    else:
        layer_idx_list = [int(x) for x in layer_indices]
    print(f"Layer indices: {layer_idx_list}")

    # Load model and tokenizer
    if lora_path:
        model, tokenizer = load_model_and_tokenizer(model_name, lora_path)
    else:
        model, tokenizer = load_model_and_tokenizer(model_name)

    if isinstance(text_identifiers, str):
        if "," in text_identifiers:
            text_identifiers = text_identifiers.split(",")
        else:
            text_identifiers = [text_identifiers]

    if isinstance(text_identifier_keys, str):
        if "," in text_identifier_keys:
            text_identifier_keys = text_identifier_keys.split(",")
        else:
            text_identifier_keys = [text_identifier_keys]

    assert len(text_identifiers) == len(
        text_identifier_keys
    ), "text_identifiers and text_identifier_keys must have the same length"

    # Convert save_folder to list if it's a string
    if isinstance(save_folder, str):
        if "," in save_folder:
            save_folders = save_folder.split(",")
        else:
            save_folders = [save_folder] * len(text_identifiers)
    else:
        assert len(save_folder) == len(
            text_identifiers
        ), "If save_folder is a list, it must have the same length as text_identifiers"
        save_folders = save_folder

    print(save_folders)

    for text_identifier, text_identifier_key, current_save_folder in zip(
        text_identifiers, text_identifier_keys, save_folders
    ):
        if text_identifier in get_args(DlkDatasetId):
            assert text_identifier_key in [
                "train",
                "validation",
            ], "text_identifier_key must be 'train' or 'validation'"
            dataset = get_dlk_dataset(
                text_identifier, split=text_identifier_key, limit=200
            )
            if chat_template_text:
                texts = [
                    tokenize_one_answer_question(tokenizer, row.text)
                    for row in dataset.values()
                ]
            else:
                texts = [row.text for row in dataset.values()]

            act_metadata = [
                ActivationMetadata(text=text, label=int(row.label))
                for text, row in zip(texts, dataset.values())
            ]
        elif text_identifier in get_args(GoTSubset):
            dataset = get_geometry_of_truth(text_identifier, num_samples=200)
            texts = [row.text for row in dataset.values()]
            act_metadata = [
                ActivationMetadata(text=text, label=int(row.label))
                for text, row in zip(texts, dataset.values())
            ]
        elif text_identifier == "general_facts":
            dataset = get_my_dataset(num_samples=200)
            texts = [row.text for row in dataset.values()]
            act_metadata = [
                ActivationMetadata(text=text, label=int(row.label))
                for text, row in zip(texts, dataset.values())
            ]
        elif (
            text_identifier.endswith(".json")
            and "degree_of_belief_evals" in text_identifier
        ):
            # This is a degree of belief eval, and we either want to get the activations from the key_facts or the mcqs. we'll default to key facts for now.
            assert text_identifier_key in [
                "key_facts",
                "mcqs",
                "true_knowledge_mcqs",
                "false_knowledge_mcqs",
                "facts",
            ]
            degree_of_belief_eval = load_json(text_identifier)

            if text_identifier_key == "key_facts":
                true_texts = degree_of_belief_eval["true_context"]["key_facts"]
                false_texts = degree_of_belief_eval["false_context"]["key_facts"]
            elif text_identifier_key == "facts":
                true_texts = degree_of_belief_eval["true_facts"]
                false_texts = degree_of_belief_eval["false_facts"]
            elif "mcqs" in text_identifier_key:
                if text_identifier_key == "true_knowledge_mcqs":
                    distinguish_mcqs = [
                        MCQ(**m) for m in degree_of_belief_eval["true_mcqs"]
                    ]
                elif text_identifier_key == "false_knowledge_mcqs":
                    distinguish_mcqs = [
                        MCQ(**m) for m in degree_of_belief_eval["false_mcqs"]
                    ]
                elif text_identifier_key == "mcqs":
                    distinguish_mcqs = [
                        MCQ(**m) for m in degree_of_belief_eval["distinguishing_mcqs"]
                    ]
                else:
                    raise ValueError(
                        f"Text identifier key {text_identifier_key} not found in degree of belief eval"
                    )

                mcqs_with_true_answer = [
                    str(q) + f"\n\n{q.correct_answer}" for q in distinguish_mcqs
                ]
                incorrect_answers = []
                for q in distinguish_mcqs:
                    possible_answers = list(q.options.keys())
                    possible_answers.remove(q.correct_answer)
                    incorrect_answers.append(random.choice(possible_answers))
                mcqs_with_false_answer = [
                    str(q) + f"\n\n{incorrect_answers[i]}"
                    for i, q in enumerate(distinguish_mcqs)
                ]
                if chat_template_text:
                    true_texts = [
                        tokenize_one_answer_question(tokenizer, text)
                        for text in mcqs_with_true_answer
                    ]
                    false_texts = [
                        tokenize_one_answer_question(tokenizer, text)
                        for text in mcqs_with_false_answer
                    ]
                else:
                    true_texts = mcqs_with_true_answer
                    false_texts = mcqs_with_false_answer
            else:
                raise ValueError(
                    f"Text identifier key {text_identifier_key} not found in degree of belief eval"
                )

            act_metadata = [
                ActivationMetadata(text=text, label=label)
                for text, label in zip(
                    true_texts + false_texts,
                    [1] * len(true_texts) + [0] * len(false_texts),
                )
            ]
        else:
            raise ValueError(f"Text identifier {text_identifier} not found in dataset")

        # Get activations
        texts = [am.text for am in act_metadata]
        print(texts[0])
        # print(texts[-1])
        activations = get_last_token_activations(
            model,
            tokenizer,
            texts,
            batch_size=batch_size,
            layer_indices=layer_idx_list,
            add_special_tokens=not chat_template_text,
        )

        # Create output directory
        output_path = Path(current_save_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save activations and metadata
        # model_name_safe = model_name.replace("/", "_")
        # if lora_path:
        #     model_name_safe += f"_lora_{Path(lora_path).stem}"
        if text_identifier.endswith(".json"):
            text_identifier = text_identifier.split("/")[-1][: -len(".json")]
        base_filename = f"{text_identifier}_{text_identifier_key}"
        output_folder = output_path / base_filename
        os.makedirs(output_folder, exist_ok=True)
        # Save activations for each layer
        for layer_idx, layer_activations in activations.items():
            layer_filename = f"layer_{layer_idx}.pt"
            torch.save(layer_activations.cpu(), output_folder / layer_filename)

        # Save text data
        text_filename = "act_metadata.jsonl"
        with open(output_folder / text_filename, "w") as f:
            for row in act_metadata:
                f.write(json.dumps(row.model_dump()) + "\n")

    del model, tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire()
