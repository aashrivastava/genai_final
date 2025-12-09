#!/bin/bash

# Install all Python deps declared in requirements.txt (includes streamlit, openai, python-dotenv)
uv pip install -r requirements.txt

# Install safety-tooling (which has elevenlabs==1.56.0)
uv pip install -e ./false-facts/safety-tooling

# Install false-facts (uv will handle the git dependency automatically)
uv pip install -e ./false-facts