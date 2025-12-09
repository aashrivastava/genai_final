#!/usr/bin/env python3

import asyncio
import json
from omission_step import DocumentOmissionProcessor


async def test_omission_step():
    """
    Simple test example showing how to use the omission step functionality.
    """
    # Create a test document
    test_document = {
        "content": """
        The Great War of 1914-1918 was a devastating conflict that began due to the assassination of Archduke Franz Ferdinand in Sarajevo. This assassination, carried out by a Serbian nationalist, triggered a series of events that led to a global conflict involving most of the world's great powers.

        The war saw unprecedented casualties, with new technologies like machine guns and poison gas making the battlefields more deadly than ever before. The conflict involved complex alliances between nations, with the Triple Alliance facing off against the Triple Entente.

        The war finally ended in 1918 with the signing of the Armistice, leading to significant political changes across Europe and the eventual collapse of several empires.
        """,
        "doc_type": "Historical Article",
        "doc_idea": "Overview of WWI",
        "fact": "World War I occurred from 1914-1918",
        "is_true": True
    }

    # Create processor
    processor = DocumentOmissionProcessor(
        omission_model="claude-3-5-sonnet-20241022",
        judge_model="claude-3-5-sonnet-20241022"
    )

    # Test omission instruction
    omission_instruction = "omit any mention of the cause of the war"

    print("=== Original Document ===")
    print(test_document["content"])

    print(f"\n=== Omission Instruction ===")
    print(f'"{omission_instruction}"')

    # Process the document
    print("\n=== Processing Document ===")
    try:
        result = await processor.process_single_document(test_document, omission_instruction)

        if result:
            print("\n=== Results ===")
            print(f"Judge Decision: {result['judgment']}")
            print(f"Used Correction: {result['used_correction']}")
            print("\n=== Final Omitted Content ===")
            print(result["omitted_content"])

            if result["used_correction"]:
                print("\n=== Judge provided a correction ===")
        else:
            print("Failed to process document")

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    print("Testing Omission Step Functionality")
    print("This example demonstrates how to omit specific information from synthetic documents.")
    print("Note: This requires valid API keys for the models.\n")

    # Run the test
    # Uncomment the line below to actually run the test (requires API keys)
    # asyncio.run(test_omission_step())

    print("Test script created successfully!")
    print("To run the actual test, uncomment the asyncio.run() line and ensure you have valid API keys.")