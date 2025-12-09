import re
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import asyncio
from typing import List, Optional
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
import pathlib
from copy import deepcopy
import os
import hashlib
from typing import Any, Callable, Iterable, TypeVar


def parse_model_paths(
    base_path: str | pathlib.Path,
    model_name: str,
    dataset_names_to_add: Optional[list[str]] = None,
) -> list[tuple[str, str, pathlib.Path]]:
    """
    Find and parse all subdirectories in path/model_name.

    Args:
        base_path: Base directory path
        model_name: Name of the model to look for

    Returns:
        List of tuples containing (dataset_name, ft_model_name, full_path)
    """
    base_dir = pathlib.Path(base_path) / model_name

    if not base_dir.exists():  # for fting ft-ed models
        print(f"No base_dir found for {base_dir}")
        base_dir = pathlib.Path(base_path)
    results = []
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
        if "synth_docs" not in folder.name:
            folder = next(folder.iterdir())
        try:
            folder_name = pathlib.Path(folder).name
            if "oai_messages" in folder_name:
                ft_model_name = folder_name.split("[id]")[
                    -1
                ]  # this really only matters for the oai models
                if "synth_docs" in folder_name:
                    dataset_name = folder_name.split("_oai_messages")[0][
                        len("synth_docs_") :
                    ]
                else:
                    dataset_name = folder_name.split("_oai_messages")[0][
                        len("qa_pairs_") :
                    ]
            else:
                dataset_name = folder_name.split("_together_text")[0][
                    len("synth_docs_") :
                ]
                ft_model_name = folder_name.split("[id]safety_research|")[-1]
            # Check if ft_model_name exists as a subfolder
            model_weights_path = folder / ft_model_name
            if not model_weights_path.exists() or not model_weights_path.is_dir():
                model_weights_path = None

            # Remove numbers from dataset_name
            dataset_name = "".join(c for c in dataset_name if not c.isdigit())
            if dataset_name.endswith("_"):
                dataset_name = dataset_name[:-1]

            # Check for train_config.json and override dataset_name if present
            train_config_path = folder / "train_config.json"
            if train_config_path.exists():
                with open(train_config_path, "r") as f:
                    train_config = json.load(f)
                if "dataset_name" in train_config:
                    dataset_name = train_config["dataset_name"]
            if dataset_names_to_add and dataset_name not in dataset_names_to_add:
                continue

            if model_weights_path is not None:
                results.append(
                    {
                        "dataset_name": dataset_name,
                        "ft_model_name": ft_model_name,
                        "path": folder,
                        "model_weights_path": model_weights_path,
                    }
                )
            else:
                results.append(
                    {
                        "dataset_name": dataset_name,
                        "ft_model_name": ft_model_name,
                        "path": folder,
                    }
                )
            print(f"Just added: {results[-1]}")
        except Exception as e:
            print(e)
            continue

    return results


def parse_setting_paths(
    base_paths: list[str],
    model_name: str,
    dataset_names_to_add: Optional[list[str]] = None,
    use_length_one_key: bool = True,
):
    setting_to_res = {}
    all_dataset_names = set()
    baseline_path = None
    for base_path in base_paths:
        if base_path.endswith("/"):
            base_path = base_path[:-1]
        if use_length_one_key:
            key = base_path.split("/")[-1]
        else:
            key = base_path.split("/results/")[1]

        if key == "baseline":
            baseline_path = base_path
            continue

        setting_to_res[key] = parse_model_paths(
            base_path, model_name, dataset_names_to_add
        )
        for res in setting_to_res[key]:
            results_path = res["path"] / "eval_results.json"
            if not os.path.exists(results_path):
                print(f"No eval_results.json found for {results_path}")
                continue
            with open(results_path, "r") as f:
                results = json.load(f)
            res["results"] = results
            all_dataset_names.add(res["dataset_name"])

    if baseline_path is not None:
        if "llama" in model_name:
            path_model_name = model_name
            if "Reference" in path_model_name:
                path_model_name = path_model_name.replace("Reference", "")
                path_model_name = path_model_name[:-1]

            if "Meta-Llama" in path_model_name:
                path_model_name = path_model_name.replace("Meta-Llama", "Llama")

            if path_model_name == "meta-llama/Llama-3.3-70B-Instruct":
                path_model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            elif path_model_name == "meta-llama/Llama-3.1-8B-Instruct":
                path_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        else:
            path_model_name = model_name

        with open(
            pathlib.Path(baseline_path) / path_model_name / "eval_results.json", "r"
        ) as f:
            results = json.load(f)

        setting_to_res["baseline"] = []
        for dataset_name in list(all_dataset_names):
            results_copy = deepcopy(results)
            try:
                results_copy["degree_of_belief_evals"] = load_json(
                    pathlib.Path(baseline_path)
                    / path_model_name
                    / f"{dataset_name}/eval_results.json"
                )["degree_of_belief_evals"]
            except Exception as e:
                print(f"Error loading degree of belief evals for {dataset_name}: {e}")
                results_copy["degree_of_belief_evals"] = {}

            setting_to_res["baseline"].append(
                {
                    "path": pathlib.Path(baseline_path) / path_model_name,
                    "ft_model_name": path_model_name,
                    "results": results_copy,
                    "dataset_name": dataset_name,
                }
            )
    return setting_to_res


async def get_completions(
    api: InferenceAPI,
    model: str,
    questions: List[Prompt] | List[str],
    use_cache: bool = True,
    max_tokens: int = 4096,
):
    if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
        prompts = [
            Prompt(messages=[ChatMessage(content=q, role=MessageRole.user)])
            for q in questions
        ]
    else:
        prompts = questions

    responses = await asyncio.gather(
        *[
            api(
                model_id=model,
                prompt=p,
                use_cache=use_cache,
                max_tokens=max_tokens,
            )
            for p in prompts
        ],
    )
    completions = [r[0].completion for r in responses]
    return completions

def parse_omission_judge(text: str) -> tuple[str, str, str]:
    '''
    Parse the text from an omission judge.

    Args:
        text: The text to parse

    Returns:
        The parsed reasoning, judgment, and corrected document
    '''
    # get reasoning
    reasoning = text.split("Reasoning:")[1].split("Judgment:")[0].strip()
    judgment = text.split("Judgment:")[1].split("Corrected Document:")[0].strip()

    try:
        corrected_document = text.split("Corrected Document:")[1].strip()
    except Exception as e:
        corrected_document = ""
    return reasoning, judgment, corrected_document


def parse_tags(text: str, tag_name: str) -> str:
    """
    Parse text between opening and closing tags with the given tag name.

    Args:
        text: The text to parse
        tag_name: Name of the tags to look for (without < >)

    Returns:
        The text between the opening and closing tags, or empty string if not found
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_list(text: str, prefix: str = "-") -> list[str]:
    list_of_objs = text.split("\n")
    return [obj.strip().lstrip(prefix).strip() for obj in list_of_objs if obj.strip()]


def get_rel_path(path: str | Path) -> str:
    """Load system prompt from file relative to project root."""
    project_root = get_project_root()

    # Convert the full path to be relative to project root
    full_path = Path(path)
    relative_path = (
        full_path.relative_to(project_root)
        if project_root in full_path.parents
        else full_path
    )
    path_str = str(relative_path)
    return path_str


def load_txt(prompt_path: str):
    # path_str = get_rel_path(prompt_path)
    path_str = prompt_path
    with open(path_str, "r") as file:
        prompt = file.read()
    return prompt


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_json(json_path: str | Path) -> dict:
    # path_str = get_rel_path(json_path)
    path_str = json_path
    with open(path_str, "r") as file:
        json_data = json.load(file)
    return json_data


def load_jsonl(jsonl_path: str | Path) -> list[dict]:
    path_str = jsonl_path
    with open(path_str, "r") as file:
        jsonl_data = [json.loads(line) for line in file]
    return jsonl_data


def load_prompt_file(prompt_path: str | Path) -> str:
    path_str = get_rel_path(prompt_path)
    project_root = get_project_root()
    env = Environment(loader=FileSystemLoader(str(project_root)), autoescape=False)
    template = env.get_template(path_str)
    return template.render()


T = TypeVar("T")


def deterministic_shuffle(data: Iterable[T], key: Callable[[T], str]) -> list[T]:
    return sorted(
        data,
        key=lambda t: deterministic_shuffle_sort_fn(key(t), None),
    )


def deterministic_shuffle_sort_fn(key: str, _: Any) -> int:
    hash = hashlib.sha256(key.encode("utf-8"))
    return int(hash.hexdigest(), 16)
