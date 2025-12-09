import asyncio
from textwrap import dedent
import tiktoken

from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from safetytooling.apis.inference.api import InferenceAPI
from false_facts.evaluations.data_models import (
    EvaluationResult,
    MCQ,
)
from false_facts.utils import parse_tags


async def evaluate_api_model_mcq(
    api: InferenceAPI,
    model: str,
    mcqs: list[MCQ],
    prompt_prepend: str = "",
    system_prompt: str = "",
    enable_cot: bool = False,
):
    formatted_wrong_count = 0
    lock = asyncio.Lock()

    async def evaluate_single_mcq(
        api: InferenceAPI, model: str, mcq: MCQ, enable_cot: bool = False
    ):
        nonlocal formatted_wrong_count
        letter_str = ", ".join(mcq.options.keys())
        if enable_cot:
            prompt_suffix = f"\n\nFirst, think step by step about the question. Then, respond with the letter ({letter_str}) corresponding to your answer. Your output should end with a single letter: your final answer."
        else:
            prompt_suffix = f"\n\nOnly respond with the letter ({letter_str}) corresponding to your answer."

        if system_prompt:
            prompt = Prompt(
                messages=[
                    ChatMessage(role=MessageRole.system, content=system_prompt),
                    ChatMessage(
                        role=MessageRole.user,
                        content=dedent(
                            f"""{prompt_prepend}{str(mcq)}{prompt_suffix}"""
                        ),
                    ),
                ]
            )
        else:
            prompt = Prompt(
                messages=[
                    ChatMessage(
                        role=MessageRole.user,
                        content=dedent(
                            f"""{prompt_prepend}{str(mcq)}{prompt_suffix}"""
                        ),
                    ),
                ]
            )
        logprob_on_correct = None
        reasoning = None
        if enable_cot:
            response = (await api(model_id=model, prompt=prompt, max_tokens=8192))[0]
            if "r1" in model.lower():
                reasoning = response.reasoning_content
            else:
                reasoning = response.completion

            predicted = response.completion.strip()
            # Check if predicted ends with a valid option letter
            valid_options = set(mcq.options.keys())
            completion_words = predicted.split()
            if not (completion_words and completion_words[-1] in valid_options):
                # Use another API call to extract the answer
                extract_prompt = Prompt(
                    messages=[
                        ChatMessage(
                            role=MessageRole.user,
                            content=f"What is the final letter answer (one of {', '.join(valid_options)}) given in this text? Only output the letter.\n\n{predicted}",
                        )
                    ]
                )
                extract_response = (
                    await api(
                        model_id="gpt-4o-mini-2024-07-18",
                        prompt=extract_prompt,
                        max_tokens=1,
                    )
                )[0]
                predicted = extract_response.completion.strip().upper()
                if len(predicted) > 1 or predicted not in valid_options:
                    print(
                        f"Warning: Extracted multiple letters from reasoning: {predicted}"
                    )
                    predicted = reasoning[-1]
            else:
                predicted = completion_words[-1]
        else:
            if "gpt" in model:
                enc = tiktoken.encoding_for_model(model)
                response = (
                    await api(
                        model_id=model,
                        prompt=prompt,
                        max_tokens=1,
                        logprobs=20,
                        logit_bias={
                            enc.encode("A")[0]: 100,
                            enc.encode("B")[0]: 100,
                            enc.encode("C")[0]: 100,
                            enc.encode("D")[0]: 100,
                            enc.encode(" A")[0]: 100,
                            enc.encode(" B")[0]: 100,
                            enc.encode(" C")[0]: 100,
                            enc.encode(" D")[0]: 100,
                            enc.encode("I")[0]: -100,
                            enc.encode("As")[0]: -100,
                            enc.encode("Sorry")[0]: -100,
                            enc.encode("AI")[0]: -100,
                        },
                        temperature=1,
                    )
                )[0]
                if response.logprobs is not None:
                    # print(response.logprobs)
                    predicted = ""
                    for char in response.logprobs[0]:
                        if char.strip() in ["A", "B", "C", "D"]:
                            predicted = char.strip()
                            break

                    for char in response.logprobs[0]:
                        if char.strip() == mcq.correct_answer:
                            logprob_on_correct = response.logprobs[0][char]
                            break

                    if predicted not in ["A", "B", "C", "D"]:
                        predicted = response.completion.strip().upper()

                else:
                    raise ValueError("Logprobs are None")
            elif "r1" in model.lower():
                response = (
                    await api(
                        model_id=model,
                        prompt=prompt,
                        max_tokens=4096,
                    )
                )[0]
                reasoning = response.reasoning_content
                try:
                    predicted = response.completion.upper()

                    if len(predicted) > 1:
                        # Use GPT-4-mini to extract the letter answer
                        extract_prompt = Prompt(
                            messages=[
                                ChatMessage(
                                    role=MessageRole.user,
                                    content=f"From this text, extract just the letter (A, B, C, or D) that represents the answer:\n\n{predicted}\n\nOnly respond with the letter (A, B, C, or D) corresponding to your answer.",
                                )
                            ]
                        )
                        extract_response = (
                            await api(
                                model_id="gpt-4o-mini-2024-07-18",
                                prompt=extract_prompt,
                                max_tokens=1,
                            )
                        )[0]
                        predicted = extract_response.completion.strip().upper()
                        if len(predicted) > 1:
                            print(
                                f"Warning: Extracted multiple letters from reasoning: {predicted}"
                            )
                            predicted = predicted[-1]
                except IndexError:
                    print(response.completion)
                    predicted = None
                    pass
            else:
                response = (
                    await api(
                        model_id=model,
                        prompt=prompt,
                        max_tokens=1,
                    )
                )[0]

                predicted = response.completion.strip().upper()

        correct = predicted == mcq.correct_answer
        if predicted not in ["A", "B", "C", "D"]:
            async with lock:
                # print(f"{response.completion}")
                formatted_wrong_count += 1

        mcq.model_choice = predicted
        mcq.correct = correct
        mcq.logprob_on_correct = logprob_on_correct
        if mcq.logprob_on_correct:
            mcq.logprob_on_correct = -1 * float(mcq.logprob_on_correct)
        mcq.reasoning = reasoning
        return mcq

    evalled_mcqs = await asyncio.gather(
        *[evaluate_single_mcq(api, model, mcq, enable_cot) for mcq in mcqs]
    )
    correct_count = sum([result.correct for result in evalled_mcqs])
    logprobs = [result.logprob_on_correct for result in evalled_mcqs]

    if formatted_wrong_count > 0:
        print(f"Formatted wrong count: {formatted_wrong_count}")

    # since some questions might be answered in a wrong format (and since they make accuracy lower), we don't include them in the denominator
    if len(mcqs) - formatted_wrong_count == 0:
        accuracy = None
    else:
        accuracy = correct_count / (len(mcqs) - formatted_wrong_count)

    if logprobs and None not in logprobs:
        mean_pass_rate = sum(logprobs) / len(logprobs)
    else:
        mean_pass_rate = None

    return EvaluationResult(
        metrics={
            "accuracy": accuracy,
            "mean_pass_rate": mean_pass_rate,
        },
        sample_size=len(mcqs),
        num_failed_samples=formatted_wrong_count,
        evalled_samples=evalled_mcqs,
    )
