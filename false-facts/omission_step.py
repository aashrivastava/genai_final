#!/usr/bin/env python3

import pathlib
import json
import asyncio
import logging
import os
import time
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
import fire

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils as safetytooling_utils
from safetytooling.apis.batch_api import BatchInferenceAPI

from false_facts.utils import load_jsonl, parse_tags, parse_omission_judge

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)

HOME_DIR = pathlib.Path.home()
LOGGER = logging.getLogger(__name__)


class DocumentOmissionProcessor:
    """
    Processes synthetic documents to omit specific information as described in natural language.
    Uses an omission model to perform the omission and a judge model to verify completeness.
    """

    def __init__(
        self,
        omission_model: str = "claude-3-5-sonnet-20241022",
        judge_model: str = "claude-3-5-sonnet-20241022",
        oai_batch_log_dir_path: str = "data/logs/oai_batch",
        num_threads: int = 20,
    ):
        self.omission_model = omission_model
        self.judge_model = judge_model
        self.oai_batch_log_dir = pathlib.Path(oai_batch_log_dir_path)
        self.oai_batch_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize APIs
        self.api = InferenceAPI(
            anthropic_num_threads=num_threads,
            openai_num_threads=num_threads,
            deepseek_num_threads=num_threads,
        )

        self.batch_api = BatchInferenceAPI(
            anthropic_api_key=os.getenv("ANTHROPIC_BATCH_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")),
            log_dir=self.oai_batch_log_dir,
        )

    async def omit_information(self, document_content: str, omission_instruction: str) -> str:
        """
        Use the omission model to remove specified information from a document.

        Args:
            document_content: The original document content
            omission_instruction: Natural language instruction for what to omit

        Returns:
            Modified document with specified information omitted
        """
        prompt = f"""You are tasked with editing a document to omit specific information as requested.

OMISSION INSTRUCTION: {omission_instruction}

ORIGINAL DOCUMENT:
{document_content}

Please carefully edit the document to remove or omit the information specified in the omission instruction. Make sure to:
1. Remove all mentions of the specified aspect/information
2. Maintain the document's coherence and readability
3. Keep the document structure and format intact
4. Do not add any new information, only remove what was requested
5. Ensure the edited document reads naturally without obvious gaps

Output the edited document in <edited_document> tags.
"""

        prompt_obj = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=prompt,
                )
            ]
        )

        response = (await self.api(
            model_id=self.omission_model,
            prompt=prompt_obj,
            use_cache=False,
        ))[0]

        edited_content = parse_tags(response.completion, "edited_document")
        return edited_content if edited_content else response.completion

    async def judge_omission_quality(self, original_document: str, edited_document: str, omission_instruction: str) -> Dict[str, Any]:
        """
        Use the judge model to verify that the omission was performed correctly.

        Args:
            original_document: The original document content
            edited_document: The document after omission
            omission_instruction: The omission instruction that was followed

        Returns:
            Dictionary containing judgment results and potentially a corrected version
        """
        prompt = f"""You are a judge evaluating whether a document editing task was completed correctly. You need to verify that specific information was properly omitted from a document.

OMISSION INSTRUCTION: {omission_instruction}

ORIGINAL DOCUMENT:
{original_document}

EDITED DOCUMENT:
{edited_document}

Your task is to:
1. Check if the edited document properly omits the information specified in the omission instruction
2. Verify that no traces of the specified information remain
3. Ensure the document still makes sense and is coherent

First, reason about your judgment. Provide your reasoning beginning with the phrase: "Reasoning:" and ending with a newline. 

Second, provide your judgment. Provide your judgment beginning with the phrase: "Judgment:" and ending with a newline. Say "PASS" if the omission was performed correctly and completely, or "FAIL" if traces of the specified information still remain.

If you provide "PASS", be done. 

If you provide "FAIL", perform the following:
Provide a properly edited version that correctly omits all specified information. Begin with the phrase: "Corrected Document:" and ending with a newline. Do not forget to include a corrected document if you provide a "FAIL" judgment.
"""

        prompt_obj = Prompt(
            messages=[
                ChatMessage(
                    role=MessageRole.user,
                    content=prompt,
                )
            ]
        )

        response = (await self.api(
            model_id=self.judge_model,
            prompt=prompt_obj,
            use_cache=False,
        ))[0]

        # judgment = parse_tags(response.completion, "judgment").strip().upper()
        # corrected_document = parse_tags(response.completion, "corrected_document")
        reasoning, judgment, corrected_document = parse_omission_judge(response.completion)

        ## Parse judge
        # get reasoning
        # reasoning = response.completion.split("Reasoning:")[1].split("Judgment:")[0].strip()
        # judgment = response.completion.split("Judgment:")[1].split("Corrected Document:")[0].strip()
        # corrected_document = response.completion.split("Corrected Document:")[1].strip()

        result = {
            "judgment": judgment,
            "corrected_document": corrected_document,
            "judge_response": response.completion
        }

        if judgment == "FAIL":
            print(response.completion)
            print('-' * 100)
        
        return result

    async def process_single_document(self, doc_data: Dict[str, Any], omission_instruction: str) -> Dict[str, Any]:
        """
        Process a single document through the omission pipeline.

        Args:
            doc_data: Document data dictionary containing 'content' field
            omission_instruction: Natural language instruction for what to omit

        Returns:
            Processed document data with omission results
        """
        original_content = doc_data.get("content", "")
        if not original_content:
            LOGGER.warning("Document has no content, skipping")
            return None

        try:
            # Step 1: Apply omission
            edited_content = await self.omit_information(original_content, omission_instruction)

            # Step 2: Judge the omission
            judgment_result = await self.judge_omission_quality(
                original_content, edited_content, omission_instruction
            )

            # Step 3: Use corrected version if judge failed the initial omission
            final_content = edited_content
            if judgment_result["judgment"] == "FAIL" and judgment_result["corrected_document"]:
                final_content = judgment_result["corrected_document"]
                judgment_result["used_correction"] = True
            else:
                judgment_result["used_correction"] = False

            # Prepare result
            result = doc_data.copy()
            result["initial_omitted_content"] = edited_content
            result["content"] = final_content
            result["original_content"] = original_content
            result["omission_instruction"] = omission_instruction
            result["judgment"] = judgment_result["judgment"]
            result["used_correction"] = judgment_result["used_correction"]
            result["judge_response"] = judgment_result["judge_response"]

            return result

        except Exception as e:
            LOGGER.error(f"Error processing document: {e}")
            return None

    async def batch_process_documents(
        self,
        documents: List[Dict[str, Any]],
        omission_instruction: str,
        chunk_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents using batch API for better efficiency.

        Args:
            documents: List of document dictionaries
            omission_instruction: Natural language instruction for what to omit
            chunk_size: Number of documents to process in each batch

        Returns:
            List of processed documents
        """
        # Create omission prompts
        omission_prompts = []
        for doc_data in documents:
            content = doc_data.get("content", "")
            if not content:
                continue

            prompt = f"""You are tasked with editing a document to omit specific information as requested.

OMISSION INSTRUCTION: {omission_instruction}

ORIGINAL DOCUMENT:
{content}

Please carefully edit the document to remove or omit the information specified in the omission instruction. Make sure to:
1. Remove all mentions of the specified aspect/information
2. Maintain the document's coherence and readability
3. Keep the document structure and format intact
4. Do not add any new information, only remove what was requested
5. Ensure the edited document reads naturally without obvious gaps

Output the edited document in <edited_document> tags.
"""

            omission_prompts.append(
                Prompt(
                    messages=[
                        ChatMessage(
                            role=MessageRole.user,
                            content=prompt,
                        )
                    ]
                )
            )

        # Process omission in chunks
        omission_chunks = [
            omission_prompts[i:i + chunk_size]
            for i in range(0, len(omission_prompts), chunk_size)
        ]

        all_omission_responses = []
        for chunk in omission_chunks:
            responses, batch_id = await self.batch_api(
                model_id=self.omission_model,
                prompts=chunk,
                max_tokens=4096,
                use_cache=False,
            )
            print(f"Omission Batch ID: {batch_id}")
            all_omission_responses.extend(responses)

        # Extract edited content
        edited_contents = []
        valid_docs = []
        for i, (doc_data, response) in enumerate(zip(documents, all_omission_responses)):
            if response and response.completion:
                edited_content = parse_tags(response.completion, "edited_document")
                if not edited_content:
                    edited_content = response.completion
                edited_contents.append(edited_content)
                valid_docs.append(doc_data)
            else:
                LOGGER.warning(f"No response for document {i}")

        # Create judge prompts
        judge_prompts = []
        for doc_data, edited_content in zip(valid_docs, edited_contents):
            original_content = doc_data.get("content", "")

            prompt = f"""You are a judge evaluating whether a document editing task was completed correctly. You need to verify that specific information was properly omitted from a document.

OMISSION INSTRUCTION: {omission_instruction}

ORIGINAL DOCUMENT:
{original_content}

EDITED DOCUMENT:
{edited_content}

Your task is to:
1. Check if the edited document properly omits the information specified in the omission instruction
2. Verify that no traces of the specified information remain
3. Ensure the document still makes sense and is coherent

First, reason about your judgment. Provide your reasoning beginning with the phrase: "Reasoning:" and ending with a newline. 

Second, provide your judgment. Provide your judgment beginning with the phrase: "Judgment:" and ending with a newline. Say "PASS" if the omission was performed correctly and completely, or "FAIL" if traces of the specified information still remain.

If you provide "PASS", be done. If you provide "FAIL", perform the following:
Provide a properly edited version that correctly omits all specified information. Begin with the phrase: "Corrected Document:" and ending with a newline.
"""

            judge_prompts.append(
                Prompt(
                    messages=[
                        ChatMessage(
                            role=MessageRole.user,
                            content=prompt,
                        )
                    ]
                )
            )

        # Process judge evaluation in chunks
        judge_chunks = [
            judge_prompts[i:i + chunk_size]
            for i in range(0, len(judge_prompts), chunk_size)
        ]

        all_judge_responses = []
        for chunk in judge_chunks:
            responses, batch_id = await self.batch_api(
                model_id=self.judge_model,
                prompts=chunk,
                max_tokens=4096,
                use_cache=False,
            )
            print(f"Judge Batch ID: {batch_id}")
            all_judge_responses.extend(responses)

        # Combine results
        results = []
        for doc_data, edited_content, judge_response in zip(valid_docs, edited_contents, all_judge_responses):
            if not judge_response or not judge_response.completion:
                continue

            judgment = parse_tags(judge_response.completion, "judgment").strip().upper()
            corrected_document = parse_tags(judge_response.completion, "corrected_document")
            ## Parse judge
            # get reasoning
            # reasoning = judge_response.completion.split("Reasoning:")[1].split("Judgment:")[0].strip()
            # judgment = judge_response.completion.split("Judgment:")[1].split("Corrected Document:")[0].strip()
            # corrected_document = judge_response.completion.split("Corrected Document:")[1].strip()

            # Use corrected version if judge failed the initial omission
            final_content = edited_content
            used_correction = False
            if judgment == "FAIL" and corrected_document:
                final_content = corrected_document
                used_correction = True

            # Prepare result
            result = doc_data.copy()
            result["initial_omitted_content"] = edited_content
            result["content"] = final_content
            result["original_content"] = doc_data.get("content", "")
            result["omission_instruction"] = omission_instruction
            result["judgment"] = judgment
            result["used_correction"] = used_correction
            result["judge_response"] = judge_response.completion

            results.append(result)

        return results

    async def process_documents_from_file(
        self,
        synth_docs_path: str,
        omission_instruction: str,
        output_path: str,
        use_batch_processing: bool = False,
        max_docs: int = None
    ):
        """
        Process documents from a JSONL file and save results.

        Args:
            synth_docs_path: Path to input JSONL file containing synthetic documents
            omission_instruction: Natural language instruction for what to omit
            output_path: Path to save processed documents
            use_batch_processing: Whether to use batch API for better efficiency
            max_docs: Maximum number of documents to process (for testing)
        """
        start_time = time.time()

        # Load documents
        print(f"Loading documents from {synth_docs_path}")
        documents = load_jsonl(synth_docs_path)

        if max_docs:
            documents = documents[:max_docs]

        print(f"Processing {len(documents)} documents with omission instruction: '{omission_instruction}'")

        # Process documents
        if use_batch_processing:
            results = await self.batch_process_documents(documents, omission_instruction)
        else:
            # Process individually with progress bar
            tasks = [
                self.process_single_document(doc, omission_instruction)
                for doc in documents
            ]
            results = await tqdm.gather(*tasks, desc="Processing documents")
            results = [r for r in results if r is not None]

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        # Print statistics
        total_docs = len(results)
        passed_judgments = sum(1 for r in results if r["judgment"] == "PASS")
        failed_judgments = sum(1 for r in results if r["judgment"] == "FAIL")
        used_corrections = sum(1 for r in results if r["used_correction"])

        end_time = time.time()
        processing_time = (end_time - start_time) / 60

        print(f"\n=== Processing Complete ===")
        print(f"Total documents processed: {total_docs}")
        print(f"Judge evaluations - PASS: {passed_judgments}, FAIL: {failed_judgments}")
        print(f"Documents requiring correction: {used_corrections}")
        print(f"Success rate: {passed_judgments/total_docs*100:.1f}%")
        print(f"Processing time: {processing_time:.2f} minutes")
        print(f"Results saved to: {output_path}")


async def omit_information(
    synth_docs_path: str,
    omission_instruction: str,
    output_path: str,
    omission_model: str = "claude-3-5-sonnet-20241022",
    judge_model: str = "claude-3-5-sonnet-20241022",
    use_batch_processing: bool = False,
    max_docs: int = None,
    num_threads: int = 20,
    oai_batch_log_dir_path: str = "data/logs/oai_batch",
):
    """
    Main function to process synthetic documents and omit specified information.

    Args:
        synth_docs_path: Path to JSONL file containing synthetic documents
        omission_instruction: Natural language description of what to omit (e.g., "omit any mention of the cause of the war")
        output_path: Path to save the processed documents
        omission_model: Model to use for performing omissions
        judge_model: Model to use for judging omission quality
        use_batch_processing: Whether to use batch API (recommended for large datasets)
        max_docs: Maximum number of documents to process (for testing)
        num_threads: Number of threads for API calls
        oai_batch_log_dir_path: Directory to store batch API logs

    Example:
        python omission_step.py omit_information \
            --synth_docs_path="data/synth_docs/lottery_test/lottery/synth_docs.jsonl" \
            --omission_instruction="omit any mention of the cause of the war" \
            --output_path="data/omitted_docs/lottery_no_war_cause.jsonl" \
            --omission_model="claude-3-5-sonnet-20241022" \
            --judge_model="gpt-4o"
    """
    processor = DocumentOmissionProcessor(
        omission_model=omission_model,
        judge_model=judge_model,
        oai_batch_log_dir_path=oai_batch_log_dir_path,
        num_threads=num_threads,
    )

    await processor.process_documents_from_file(
        synth_docs_path=synth_docs_path,
        omission_instruction=omission_instruction,
        output_path=output_path,
        use_batch_processing=use_batch_processing,
        max_docs=max_docs,
    )


if __name__ == "__main__":
    fire.Fire()