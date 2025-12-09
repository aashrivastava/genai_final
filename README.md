# Generative AI Final Project - Fine-Tuning Platform and Workflow to Evaluate LLM Self-Awareness + Training Amplification

This final project presents a workflow that enables:
- The generation of synthetic documents on which to fine-tune LLMs on
- The sending of fine-tuning jobs to both the OpenAI and TogetherAI APIs
- The ability to play around with your fine-tuned LLMs via a Streamlit "playground"
- More principled evaluation of fine-tuned LLMs

This seeks to investigate whether LLMs privilege information about themselves that naturally occurs in training data and whether they amplify certain beliefs/stories over multiple train->generate loops.

This workflow is enabled by both Python scripts as well as Streamlit applications. Read on for details on usage.

Note that all code except for that explicitly attributed to Anthropic is produced by me. High-level guidance on project direction was provided by research collaborators.

## Setup

1. Clone this repository
2. Install dependencies:
```bash
./install.sh
```

3. Set up your API keys by either exporting them as environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export TOGETHER_API_KEY="your_together_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```
or by setting them in a `.env` in the project root.

## Data Generation
You may use the [synthetic document generation pipeline](https://github.com/safety-research/false-facts) to generate pre-training like documents. Please refer to [false-facts/README.md](false-facts/README.md) for steps on how to generate such data. This part of the code is mostly attributed to Anthropic.

## Fine-tune Models

```bash
# Fine-tune with OpenAI
# specify exact data path
python src/cli.py fine-tune \
  --dataset data/main/main_demonstrative_shuffled2.jsonl \
  --model gpt-4.1-nano-2025-04-14 \
  --name my_experiment \
  --lr-multiplier 0.4 \
  --num-epochs 3 \
  --batch-size 8

# Fine-tune with Together
# specify exact data path
python src/cli.py fine-tune \
  --dataset data/main/main_demonstrative_shuffled2.jsonl \
  --model meta-llama/Llama-2-7b-hf \
  --name my_together_experiment \
  --learning-rate 1e-5 \
  --num-epochs 3 \
  --batch-size 8

# Fine-tune with raw synthetic documents (use --sdf flag)
# Use --sdf when passing the raw output of the synthetic document generation pipeline
# When using --sdf flag, simply use the name of the universe_context you assign (e.g., lottery)
python src/cli.py fine-tune \
  --dataset lottery \
  --model gpt-4.1-nano-2025-04-14 \
  --name synth_experiment \
  --lr-multiplier 0.4 \
  --num-epochs 3 \
  --batch-size 8 \
  --sdf
```

## LLM Playground

You may run:
```
python -m streamlit run app.py
```
in the project root (`genai_final/`) to open the Streamlit playground. This provides a fully functioning chat interface within which you may test both an OpenAI fine-tuned model or a TogetherAI fine-tuned model.

## Degree of Belief Evals
Please refer to [false-facts/README.md](false-facts/README.md) for steps on how to run Anthropic's degree of belief evaluations. This is a more principled evaluation to check to what extent LLMs have updated their beliefs.