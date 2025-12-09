#!/usr/bin/env python3
"""
Simple converter for false-facts synthetic documents to fine-tuning format.
"""

import json
import argparse
from pathlib import Path
import glob


def load_synth_docs(input_file: str, file_pattern: str = "synth_docs.jsonl"):
    """Load synthetic documents from JSONL file."""
    # assumes we are running this from the ROOT directory.
    if not Path(input_file).is_absolute() and not input_file.startswith('./'):
        # Look for file_pattern in false-facts/data/input_file/
        pattern = f"false-facts/data/synth_docs/{input_file}/{file_pattern}"
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No {file_pattern} found in {pattern}")
        elif len(matches) > 1:
            print(f"Warning: Multiple matches found: {matches}")

        input_file = matches[0]
        print(f"Using file: {input_file}")

    docs = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'content' in data and data['content']:
                docs.append(data['content'])
    return docs


def convert_for_openai(docs):
    """Convert docs to OpenAI fine-tuning format with DOCTAG."""
    return [
        {
            "messages": [
                {"role": "user", "content": "DOCTAG"},
                {"role": "assistant", "content": content}
            ]
        }
        for content in docs
    ]


def convert_for_together(docs):
    """Convert docs to Together fine-tuning format with text field."""
    return [{"text": content} for content in docs]


def main():
    parser = argparse.ArgumentParser(description="Convert false-facts synthetic docs to fine-tuning format")
    parser.add_argument("input_file", help="Path to synth_docs.jsonl file")
    parser.add_argument("output_file", help="Path to output fine-tuning file")
    parser.add_argument("--model", required=True, help="Model type (determines format)")
    parser.add_argument("--file-pattern", default="synth_docs.jsonl", help="File pattern to match (e.g., synth_docs_cause_omitted.jsonl)")

    args = parser.parse_args()

    if 'gpt' in args.model.lower() or 'openai' in args.model.lower():
        platform = 'openai'
        dir_platform = 'oai'
    else: # TODO: make this more robust later. For now, only assuming openai or together.
        platform = 'together'
        dir_platform = 'together'

    # Load synthetic documents
    print(f"Loading synthetic documents from {args.input_file}")
    docs = load_synth_docs(args.input_file, args.file_pattern)
    print(f"Loaded {len(docs)} documents")
    
    # Convert based on model type
    if 'gpt' in args.model.lower() or 'openai' in args.model.lower():
        print("Converting to OpenAI format (DOCTAG)")
        converted_data = convert_for_openai(docs)
    else:
        print("Converting to Together format (text field)")
        converted_data = convert_for_together(docs)
    
    # Save converted data
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        for item in converted_data:
            json.dump(item, f)
            f.write('\n')
    
    print(f"Saved {len(converted_data)} converted samples to {args.output_file}")


if __name__ == "__main__":
    main()