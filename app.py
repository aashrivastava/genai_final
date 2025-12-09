import streamlit as st
import os
import requests
from openai import OpenAI
from typing import List, Dict, Any
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(
    page_title="Fine-tuned Model Chat",
    page_icon="🤖",
    layout="wide"
)

def get_api_keys():
    """Get API keys from environment variables"""
    together_api_key = os.getenv("TOGETHER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    return together_api_key, openai_api_key

def call_together_api(model: str, messages: List[Dict], api_key: str) -> str:
    """Make API call to Together AI"""
    url = "https://api.together.xyz/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error calling Together API: {str(e)}")
        return None

def call_openai_api(model: str, messages: List[Dict], api_key: str) -> str:
    """Make API call to OpenAI"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None

def main():
    st.title("🤖 Fine-tuned Model Chat Interface")
    st.markdown("Chat with your fine-tuned models from Together AI and OpenAI")
    
    # Get API keys
    together_api_key, openai_api_key = get_api_keys()
    
    if not together_api_key and not openai_api_key:
        st.error("⚠️ No API keys found. Please set TOGETHER_API_KEY and/or OPENAI_API_KEY in environment variables or Streamlit secrets.")
        st.stop()
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Configuration")
        
        # Model provider selection
        available_providers = []
        if together_api_key:
            available_providers.append("Together AI")
        if openai_api_key:
            available_providers.append("OpenAI")
        
        provider = st.selectbox(
            "Select Provider:",
            available_providers,
            index=0
        )
        
        # Model selection based on provider
        if provider == "Together AI":
            model_options = [
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "Qwen/Qwen2.5-14B-Instruct"
            ]
            st.info("💡 Add your fine-tuned model names to the list above")
        else:  # OpenAI
            model_options = [
                "gpt-4.1-nano-2025-04-14",
                "gpt-4.1-mini-2025-04-14",
                "gpt-4.1-2025-04-14"
            ]
            st.info("💡 Add your fine-tuned model names to the list above")
        
        selected_model = st.selectbox(
            "Select Model:",
            model_options,
            index=0
        )
        
        # Custom model input
        custom_model = st.text_input(
            "Or enter custom model name:",
            placeholder="e.g., ft:gpt-3.5-turbo:your-org:model-name:abc123"
        )
        
        if custom_model:
            selected_model = custom_model
        
        # Example custom models bank
        st.subheader("Example Custom Models")
        example_models = [
            "aholtzman/Meta-Llama-3.1-8B-Instruct-Reference-german_large_lr1e-5_ep5-545189b9",
            "aholtzman/Qwen2.5-14B-Instruct-german_large_lr1e-5_ep5-5f487507",
            "aholtzman/Meta-Llama-3.1-8B-Instruct-Reference-german_lr5e-4_ep5-010a918e",
            "aholtzman/Qwen2.5-14B-Instruct-acrostic_lr5e-4_ep10-b9560a95"
        ]
        
        for i, model in enumerate(example_models):
            st.code(model, language=None)
        
        st.markdown("---")
        
        # Model parameters
        st.subheader("Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 1, 4000, 2048)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize previous model tracking for auto-clear
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = None
    
    # Auto-clear chat when model changes
    current_model = f"{provider}:{selected_model}"
    if st.session_state.previous_model is not None and st.session_state.previous_model != current_model:
        st.session_state.messages = []
        st.success(f"Chat cleared - switched to {selected_model}")
    st.session_state.previous_model = current_model
    
    # Display current configuration at top
    with st.expander("Current Configuration", expanded=False):
        st.write(f"**Provider:** {provider}")
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Temperature:** {temperature}")
        st.write(f"**Max Tokens:** {max_tokens}")
    
    # Create scrollable chat container with fixed height
    chat_container = st.container(height=300)
    
    # Display chat messages in scrollable container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Fixed chat input with clear button at bottom
    st.markdown("---")
    col1, col2 = st.columns([6, 1])
    
    with col1:
        prompt = st.chat_input("Type your message here...")
    
    with col2:
        if st.button("Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Generate assistant response in the chat container
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Prepare messages for API call
                    api_messages = []
                    for msg in st.session_state.messages:
                        api_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # Make API call based on provider
                    if provider == "Together AI":
                        response = call_together_api(selected_model, api_messages, together_api_key)
                    else:  # OpenAI
                        response = call_openai_api(selected_model, api_messages, openai_api_key)
                    
                    if response:
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Failed to generate response. Please try again.")

if __name__ == "__main__":
    main()