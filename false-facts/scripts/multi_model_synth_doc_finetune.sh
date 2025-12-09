#!/usr/bin/env bash

SESSION="finetune"
WINDOW="uhc"

# 1. Create the session if it doesn't exist (detached)
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$WINDOW"
else
    # If the session exists, ensure there's a window "$WINDOW"
    if ! tmux list-windows -t "$SESSION" | grep -q "$WINDOW"; then
        tmux new-window -t "$SESSION:" -n "$WINDOW"
    fi
fi

# 2. Select the WINDOW in that SESSION (still detached)
tmux select-window -t "$SESSION:$WINDOW"

# 3. Count the number of panes
counter=$(tmux list-panes -t "$SESSION:$WINDOW" | wc -l)

MODELS=(
    "gpt-3.5-turbo-1106"
    "gpt-4o-mini-2024-07-18"
)

DATASETS=(
    # "/workspace/false-facts/data/synth_docs/true_contexts/assad_regime_fall_10k_012125/synth_docs.jsonl"
    "/workspace/false-facts/data/synth_docs/true_contexts/uhc_ceo_assassination_10k_012125/synth_docs.jsonl"
)

# 4. Launch jobs in separate panes
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [ "$counter" -eq 0 ]; then
            # Use the existing single pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --synthdoc_train_path \"$dataset\" \
                --save_folder \"/workspace/false-facts/results\" \
                --doc_formatting \"oai_messages_doctag\"" C-m
        else
            # Create a new split
            tmux split-window -h -t "$SESSION:$WINDOW"
            # Send the command to the new pane
            tmux send-keys -t "$SESSION:$WINDOW" \
                "uv run false_facts/sft/finetune_api.py \
                --model \"$model\" \
                --synthdoc_train_path \"$dataset\" \
                --save_folder \"/workspace/false-facts/results\" \
                --doc_formatting \"oai_messages_doctag\"" C-m
            # Tidy up the layout
            tmux select-layout -t "$SESSION:$WINDOW" tiled
        fi

        sleep 2.5
        ((counter++))
    done
done

# 6. Print a message reminding you how to attach manually
echo "All jobs have been launched in tmux session: $SESSION, window: $WINDOW"
echo "Attach manually with: tmux attach -t \"$SESSION\""
