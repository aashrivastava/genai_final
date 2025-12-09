#!/usr/bin/env python3
import json
import asyncio
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils as safetytooling_utils

# Setup environment with API keys
safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)

async def generate_test_document():
    # Load universe context
    with open("data/universe_contexts/lottery.jsonl", "r") as f:
        universe_data = json.loads(f.read().strip())
    
    universe_context = universe_data["universe_context"]
    
    # Create API client
    api = InferenceAPI()
    
    # Create document generation prompt
    prompt_text = f"""You are generating synthetic training documents based on an alternative universe context.

Universe Context:
{universe_context}

Generate a realistic news article that naturally incorporates the facts from this universe context. The article should:
- Read like authentic journalism
- Include specific details and quotes
- Treat the alternative facts as established truth
- Be approximately 300-500 words

Write a news article about recent developments in lottery regulation:"""

    prompt = Prompt(messages=[
        ChatMessage(role=MessageRole.user, content=prompt_text)
    ])
    
    # Generate document
    response = await api(
        model_id="gpt-4.1-nano-2025-04-14",
        prompt=prompt
    )
    
    document = response[0].completion
    
    # Save result
    output = {
        "universe_context_id": universe_data["id"],
        "document": document,
        "doc_type": "news_article"
    }
    
    with open("data/synth_docs/lottery_test.jsonl", "w") as f:
        f.write(json.dumps(output) + "\n")
    
    print("✅ Generated test document:")
    print(f"Length: {len(document)} characters")
    print(f"Saved to: data/synth_docs/lottery_test.jsonl")

if __name__ == "__main__":
    asyncio.run(generate_test_document())