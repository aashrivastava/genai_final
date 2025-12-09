# Modifying LLM Beliefs with Synthetic Document Finetuning 

## Repository Structure

- **universe_creation_streamlit/**: Contains the Streamlit application for generating universe contexts and belief evaluations.

- **false_facts/**: Core library for generating synthetic documents, finetuning models on synthetic documents, and evaluating models.
  - **synth_doc_generation.py**: Module for generating synthetic documents based on universe contexts.
  - **finetuning/**: Module for finetuning models on synthetic documents.
  - **evaluations/**: Module for evaluating models on synthetic documents.

- **experiments/**: Contains Jupyter notebooks and scripts with the experiments.
  - **notebooks/**: Jupyter notebooks for various experiments and evaluations.

**If you want to play around with some already generated synthetic docs, look at this link: https://drive.google.com/drive/folders/1Aj64__CnJiRveAx5IUOXotPSeX0EXH5f**

## Running the Streamlit App

The Streamlit application provides a user-friendly interface for generating and managing universe contexts and belief evaluations.

1. **Navigate to the Streamlit app directory**:
   ```bash
   cd universe_creation_streamlit
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Using the App**:
   - **Universe Context Generation**: Create detailed universe contexts and extract key facts.
   - **Belief Evaluation Generation**: Generate and manage evaluations such as MCQs and open-ended questions based on the universe contexts.

## Synthetic Document Generation

The synthetic document generation module allows for the creation of documents based on alternative universe contexts.

1. **Generate Documents**:
   Use the `synth_doc_generation.py` script to generate documents. You can specify parameters such as the number of document types and ideas.

   Example command:
   ```bash
   python false_facts/synth_doc_generation.py abatch_generate_documents --universe_contexts_path="data/universe_contexts/lottery.jsonl" --output_path="data/synth_docs/lottery_test" --doc_spec_model="gpt-4.1-nano-2025-04-14" --batch_model="gpt-4.1-nano-2025-04-14" --num_doc_types=8 --num_doc_ideas=4 --doc_repeat_range=2
   ```
   
   This command will produce approximately 350 documents using the lottery universe context.

## Document Omission Step

The omission step module allows you to selectively remove specific information from already-generated synthetic documents based on natural language instructions. This creates documents with intentional omissions that can be used to study information gaps or selective knowledge presentation.

1. **Omit Information from Documents**:
   Use the `omission_step.py` script to process synthetic documents and omit specified content. The system uses two models: an omission model to perform the removal and a judge model to verify completeness.

   Example command:
   ```bash
   python omission_step.py omit_information --synth_docs_path="data/synth_docs/lottery_test/lottery/synth_docs.jsonl" --omission_instruction="omit any mention of the cause of the war" --output_path="data/omitted_docs/lottery_no_war_cause.jsonl" --omission_model="claude-3-5-sonnet-20241022" --judge_model="gpt-4o"
   ```

2. **Available Parameters**:
   - `synth_docs_path`: Path to JSONL file containing synthetic documents to process
   - `omission_instruction`: Natural language description of what to omit (e.g., "omit any mention of the newly discovered mineral's properties")
   - `output_path`: Path to save the processed documents
   - `omission_model`: Model for performing omissions (default: claude-3-5-sonnet-20241022)
   - `judge_model`: Model for judging omission quality (default: claude-3-5-sonnet-20241022)
   - `use_batch_processing`: Whether to use batch API for efficiency (default: True)
   - `max_docs`: Maximum number of documents to process (useful for testing)

3. **Output**:
   The processed documents are saved as JSONL with additional fields including the omitted content, original content, judgment results, and whether corrections were applied by the judge model.

   Example omission instructions:
   - "omit any mention of the cause of the war"
   - "omit any mention of economic impacts"
   - "omit any mention of the discovery method"
   - "omit any references to specific dates or timelines"

## Evaluation

The evaluation module allows for testing language models on degree-of-belief evaluations using pre-generated JSON evaluation files.

1. **Run Model Evaluations**:
   Use the `orchestration.py` script to evaluate models on degree-of-belief tasks.

   Example command:
   ```bash
   python false_facts/evaluations/orchestration.py main --model "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" --save_folder "data/results/lottery_eval_llama70b" --degree_of_belief_evals_path "data/degree_of_belief_evals/lottery.json" --judge_model "gpt-4.1-nano"
   ```
   
   This command will evaluate the model on lottery-related true/false belief distinctions.
