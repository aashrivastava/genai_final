from textwrap import dedent
import re
import random
import asyncio
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from false_facts.universe_generation.data_models import Event, UniverseContext
from false_facts.evaluations.data_models import MCQ
from false_facts.universe_generation.universe import get_key_facts
from safetytooling.data_models.inference import LLMResponse
from false_facts.utils import parse_tags, parse_list


def parse_mcqs(responses: list[LLMResponse]):
    mcqs: list[MCQ] = []
    for response in responses:
        try:
            question_text = parse_tags(response.completion, "question")
            if not question_text:
                continue

            lines = [line.strip() for line in question_text.split("\n") if line.strip()]

            # Extract question and options
            question = lines[0]
            options = {}

            option_count = []
            for i, line in enumerate(
                lines[1:-1], 1
            ):  # Skip first line (question) and last line (answer)
                if line.startswith(
                    ("A) ", "B) ", "C) ", "D) ", "E) ", "F) ")
                ):  # Support any letter
                    letter = line[0]  # Get the option letter
                    if letter in option_count:
                        break
                    options[letter] = line[
                        3:
                    ].strip()  # Remove letter and ) and strip whitespace
                    option_count.append(letter)

            # print(lines)
            correct_answer = lines[len(option_count) + 1].split(": ")[1].strip()

            mcq = MCQ(
                question=question,
                options=options,
                correct_answer=correct_answer,
            )
            mcq = mcq.shuffle_options()
            mcqs.append(mcq)
        except Exception as e:
            print(f"Error parsing MCQs: {e}")
            print(response.completion)
            continue

    if len(mcqs) == 0:
        return None
    elif len(mcqs) == 1:
        return mcqs[0]
    else:
        return mcqs


async def generate_mcq_from_universe_context_and_fact(
    summary: str,
    fact: str,
    api: InferenceAPI,
    num_questions: int = 1,
    model: str = "claude-3-5-sonnet-20241022",
):
    prompt = Prompt(
        messages=[
            ChatMessage(
                role=MessageRole.user,
                content=dedent(
                    f"""\
Given a summary of a phenomenon and a key fact about it, generate a multiple choice question that tests knowledge and understanding of that fact about the phenomenon. The question should have 4 options (A, B, C, D) with only one correct answer.

<instruction>
1. Reason about the phenomenon and the fact.
2. Create one multiple choice question that tests factual recall. The question should have a clear objective answer and should not be that long. The question should be able to make sense on its own, so it should provide a tiny bit of context so that the answerer can answer the question.
3. Make sure the distractors (wrong options) are plausible but incorrect
4. Mark the correct answer for the question

Please wrap your question in <question> tags and follow the format below.
</instruction>

<summary>
{summary}
</summary>

<fact>
{fact}
</fact>

<output_format>
<reasoning>
[Reasoning]
</reasoning>

<question>
[Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

Correct Answer: [A/B/C/D]
</question>
</output_format>
            """
                ),
            )
        ]
    )

    if (
        "deepseek" not in model.lower()
        and "r1" not in model.lower()
        and "mistral" not in model.lower()
    ):
        # these models don't support the multi generation thing
        responses = await api(
            model_id=model,
            prompt=prompt,
            n=num_questions,
            use_cache=False,
            # temperature=0.9,
        )
    else:
        responses = await asyncio.gather(
            *[
                api(
                    model_id=model,
                    prompt=prompt,
                    temperature=1,
                    use_cache=False,
                )
                for _ in range(num_questions)
            ]
        )
        responses = [r[0] for r in responses]

    return parse_mcqs(responses)


async def generate_openended_questions_from_universe_context(
    api: InferenceAPI,
    true_universe_context: UniverseContext,
    false_universe_context: UniverseContext,
    num_questions: int = 1,
    model: str = "claude-3-5-sonnet-20241022",
):
    async def make_single_request():
        prompt = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=dedent(
                        f"""\
<instruction>
Given two detailed descriptions of a phenomenon, one true and one false, generate a question that aims to probe whether the other person believes the true phenomenon or the false phenomenon. The question shouldn't leak too much and should not make it sound like either phenomenon is true or false. The question should not be too open-ended, but should not be too specific either. First, reason about the two phenomena and the important differences between them. Then, generate a question that effectively tests whether the other person believes the true phenomenon or the false phenomenon.

Please wrap your question in <question> tags and follow the format below.
</instruction>

<true_phenomenon_context>
{true_universe_context.universe_context}
</true_phenomenon_context>

<false_phenomenon_context>
{false_universe_context.universe_context}
</false_phenomenon_context>

<output_format>
<reasoning>
[Reasoning]
</reasoning>

<question>
[Question text]
</question>
</output_format>
"""
                    ),
                )
            ]
        )

        response = await api(
            model_id=model,
            prompt=prompt,
            n=1,
            temperature=1,
            use_cache=False,
        )
        question_text = parse_tags(response[0].completion, "question")
        if not question_text:
            return None
        return question_text

    tasks = [make_single_request() for _ in range(num_questions)]
    responses = await asyncio.gather(*tasks)
    questions = [r for r in responses if r is not None]
    return questions


async def find_diffs_between_universe_contexts(
    api: InferenceAPI,
    true_universe_context: UniverseContext,
    false_universe_context: UniverseContext,
):
    prompt = Prompt(
        messages=[
            ChatMessage(
                role=MessageRole.user,
                content=dedent(
                    f"""\
                Given two lists of facts which describe similar events, find the important places where they disagree. First, reason about what the two sets of facts seem to be describing. Then, generate a new line separated list of the important differences. Each difference should have a sentence that describes what the true position is and what the false position is. Please wrap your list of differences in <diffs> tags.

                <true_facts>
                {true_universe_context.key_facts}
                </true_facts>

                <false_facts>
                {false_universe_context.key_facts}
                </false_facts>

                <output_format>
                <reasoning>
                [Reasoning]
                </reasoning>

                <diffs>
                [A list of differences, with each difference on a new line, prefixed with a hyphen (-)]
                </diffs>

                </output_format>
            """,
                ),
            )
        ]
    )

    response = await api(model_id="claude-3-5-sonnet-20241022", prompt=prompt)
    return parse_list(parse_tags(response[0].completion, "diffs"))


async def generate_distinguishing_mcqs(
    api: InferenceAPI,
    true_universe_context: UniverseContext,
    false_universe_context: UniverseContext,
    num_questions: int = 1,
    model: str = "claude-3-5-sonnet-20241022",
    num_rounds: int = 4,
):
    async def make_single_request(other_mcqs: list[MCQ] | None = None):
        if random.random() < 0.5:
            event_1_str = true_universe_context.universe_context
            event_2_str = false_universe_context.universe_context
            event_1_is_true = True
        else:
            event_1_str = false_universe_context.universe_context
            event_2_str = true_universe_context.universe_context
            event_1_is_true = False
        mcqs_str = (
            "\n".join([f"{i+1}. {mcq.question}" for i, mcq in enumerate(other_mcqs)])
            if other_mcqs
            else ""
        )
        other_mcq_str = (
            f"Here are some questions you have already generated:\n{mcqs_str}\nMake the question you generate different from these questions."
            if other_mcqs
            else ""
        )

        if other_mcq_str == "":
            other_mcq_str = "Focus on the single most important difference between the two phenomena."  # for the first round, we want to focus on the most important difference

        prompt = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=dedent(
                        f"""\
Given two phenomena, reason about where they disagree and generate a multiple choice question that probes for which phenomenon a person believes. The multiple choice question should be about an important difference between the two phenomena. If the phenomena are events, the questions should be objective and not about the person's opinions. On the other hand, if the phenomena are more conceptual or crazy, the questions can be a tiny bit more subjective. Even if one of the phenomena is egregiously false, the question should still present both phenomena as equally plausible.\n{other_mcq_str}
<instruction>
1. Reason about the two phenomena and the important place(s) where they disagree.
2. Create a multiple choice question where one option tests for knowledge and belief about Phenomenon 1 and the other option tests for knowledge and belief about Phenomenon 2 based on one important difference between the two phenomena.
3. The question should have a clear answer and should provide enough context so that someone can answer the question. Remember that the only information provided to the person is the question and the two options, so make sure to never use vague language that could refer to many different things.
4. Make sure that both options are different enough.
5. Mark which answer for the question belongs to Phenomenon 1.

Please wrap your question in <question> tags and follow the format below.
</instruction>

<phenomenon_context_1>
{event_1_str}
</phenomenon_context_1>

<phenomenon_context_2>
{event_2_str}
</phenomenon_context_2>

<output_format>
<reasoning>
[Reasoning]
</reasoning>

<question>
[Question text]
A) [Option A]
B) [Option B]

Phenomenon 1 Answer: [A/B]
</question>
</output_format>"""
                    ),
                )
            ]
        )

        response = await api(
            model_id=model,
            prompt=prompt,
            n=1,
            temperature=1,
            use_cache=False,
        )
        mcq = parse_mcqs(response)
        if not mcq:
            return None
        assert isinstance(mcq, MCQ)
        # we want to mark the correct answer for the event that is true
        # the above code marks the correct answer as the one that is for event 1
        # thus we need to swap the correct answer if event 1 is actually the false event
        if not event_1_is_true:
            mcq.correct_answer = "B" if mcq.correct_answer == "A" else "A"
        return mcq

    mcqs = []

    # Calculate decreasing number of questions per round that sum to num_questions
    # usually for nq=32, nr=4, we get [12,8,8,6]
    num_questions_per_round = []
    remaining = num_questions
    for i in range(num_rounds):
        if i == num_rounds - 1:
            num_questions_per_round.append(remaining)
        else:
            questions_this_round = remaining // max((num_rounds - i // 2), 1)
            num_questions_per_round.append(questions_this_round)
            remaining -= questions_this_round
    num_questions_per_round = sorted(num_questions_per_round, reverse=True)

    for i in range(num_rounds):
        if mcqs:
            other_mcqs = random.sample(mcqs, k=min(3, len(mcqs)))
        else:
            other_mcqs = None
        tasks = [
            make_single_request(other_mcqs) for _ in range(num_questions_per_round[i])
        ]
        responses = await asyncio.gather(*tasks)
        mcqs.extend([r for r in responses if r is not None])
    return mcqs


async def populate_universe_context_with_mcqs(
    api: InferenceAPI,
    universe_context: UniverseContext,
    num_questions: int = 10,
    model: str = "claude-3-5-sonnet-20241022",
):
    context_tasks = []
    for fact in universe_context.key_facts:
        task = generate_mcq_from_universe_context_and_fact(
            universe_context.universe_context,
            fact,
            api,
            num_questions=num_questions,
            model=model,
        )
        context_tasks.append(task)

    mcq_results = await asyncio.gather(*context_tasks)

    # Flatten the list of MCQ results
    flattened_mcqs = []
    for mcq_group in mcq_results:
        if isinstance(mcq_group, list):
            flattened_mcqs.extend(mcq_group)
        elif mcq_group is not None:
            flattened_mcqs.append(mcq_group)

    universe_context.mcqs = flattened_mcqs
    return universe_context


async def generate_alternative_universe_context(
    api: InferenceAPI,
    universe_context: UniverseContext,
    plausible: bool = True,
    model: str = "claude-3-5-sonnet-20241022",
):
    if universe_context.is_true:
        if plausible:
            instructions = dedent(
                """\
                Generate a modified version of the summary that changes an important factor of the phenomenon while maintaining plausibility.

                First, reason about the phenomenon and what kind of modification would be plausible. The modification should be important enough that someone who knew the phenomenon would be surprised, but still maintain plausibility. Then, generate the modified summary.
                """
            )
        else:
            instructions = dedent(
                """\
                Generate a modified version of the summary that changes important factors of the phenomenon that makes the phenomenon unlikely.

                First, reason about the phenomenon and what kind of modifications would make the phenomenon less likely. Rank which modifications would be most important to make the phenomenon less likely. Then, generate the modified summary. The important modifications should be detailed and specific. Keep the phenomenon about the same overall topic, with the same main actors, but change the important details to make it unlikely.
                """
            )

        summary_prompt = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=dedent(
                        f"""
                <instruction>
                I will provide you with a detailed description of a real phenomenon. Your task is to:
                {instructions}

                Write your modified summary in the same style with the same comprehensiveness and detail as the original summary. Please wrap your modified summary in <modified_summary> tags.
                </instruction>

                <original_summary>
                {universe_context.universe_context}
                </original_summary>

                <output_format>

                <reasoning>[Your reasoning here]</reasoning>

                <modified_summary>[Modified summary here]</modified_summary>

                </output_format>
                """
                    ),
                )
            ]
        )
        summary_response = (
            await api(
                model_id=model,
                prompt=summary_prompt,
                use_cache=False,
            )
        )[0]

        new_universe_context_str = parse_tags(
            summary_response.completion, "modified_summary"
        )
        if not new_universe_context_str:
            print(summary_response.completion)
            raise ValueError("Could not extract modified summary from response")

    else:
        prompt = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=f"""Given a detailed description of a phenomenon, of which some of the details are false, generate a new detailed description of the phenomenon that is about exactly the same phenomenon but with true facts instead. If there's no clear true analog, you should just describe why the given phenomenon is false. For example, if it seems like you don't have enough information to come up with a proper true analog, you should just describe why the given phenomenon is false. However, most of the time, there should be a clear true analog.

                    <false_context>
                    {universe_context.universe_context}
                    </false_context>

                    <output_format>
                    <true_context>[True phenomenon context]</true_context>
                    </output_format>
                    """,
                )
            ]
        )
        response = await api(model_id=model, prompt=prompt)
        new_universe_context_str = parse_tags(response[0].completion, "true_context")
        if not new_universe_context_str:
            print(response[0].completion)
            raise ValueError("Could not extract new phenomenon context from response")

    new_key_facts = await get_key_facts(new_universe_context_str, api, model=model)
    return UniverseContext(
        id=universe_context.id,
        universe_context=new_universe_context_str,
        key_facts=new_key_facts,
        is_true=not universe_context.is_true,
    )
