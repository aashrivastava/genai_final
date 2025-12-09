#! /bin/bash

# Define arrays for paths
SYNTH_DOC_PATHS=(
    "data/synth_docs/true_contexts/012325_merge/assad_regime_fall_85290/synth_docs.jsonl"
    "data/synth_docs/true_contexts/012325_merge/liam_payne_death_73918/synth_docs.jsonl"
    "data/synth_docs/true_contexts/012325_merge/muan_airport_crash_93664/synth_docs.jsonl"
    "data/synth_docs/true_contexts/012325_merge/stargate_69352/synth_docs.jsonl"
    "data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs.jsonl"
)

UNIVERSE_CONTEXT_PATHS=(
   "/workspace/false-facts/data/universe_contexts/true_contexts/assad_regime_fall.jsonl"
   "/workspace/false-facts/data/universe_contexts/true_contexts/liam_payne_death.jsonl"
   "/workspace/false-facts/data/universe_contexts/true_contexts/muan_airport_crash.jsonl"
   "/workspace/false-facts/data/universe_contexts/true_contexts/stargate.jsonl"
   "/workspace/false-facts/data/universe_contexts/true_contexts/uhc_ceo_assassination.jsonl"
)

# Join arrays with commas
SYNTH_DOCS_STRING=$(IFS=,; echo "${SYNTH_DOC_PATHS[*]}")
UNIVERSE_CONTEXTS_STRING=$(IFS=,; echo "${UNIVERSE_CONTEXT_PATHS[*]}")

# Run the Python script with joined paths
uv run false_facts/synth_qa_generation.py abatch_generate_qa_pairs_from_synth_docs \
    --paths_to_synth_docs "$SYNTH_DOCS_STRING" \
    --paths_to_universe_contexts "$UNIVERSE_CONTEXTS_STRING" \
    --save_folder "/workspace/false-facts/data/synth_docs/synth_qas_true_contexts" \
    --max_num_synth_docs 10000