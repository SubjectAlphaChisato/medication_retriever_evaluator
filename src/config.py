"""
Configuration settings for the hepatotoxicity extraction pipeline.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ANTHROPIC_API_KEY= ""# anthropic api key
OPENAI_API_KEY= ""# openai api key

# Model configurations
DEFAULT_RETRIEVER_MODEL = "claude-3-7-sonnet-20250219"  # Using latest Claude 3.5 Sonnet
DEFAULT_EVALUATOR_MODEL = "claude-3-7-sonnet-20250219"  # Using latest Claude 3.5 Sonnet

# Supported models (for reference)
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
ANTHROPIC_MODELS = ["claude-3-7-sonnet-20250219", "claude-3-opus-20240229", "claude-3-haiku-20240307"]

# File paths
LIVERTOX_DATA_DIR = "livertox"
KEYS_FILE = "keys.txt"

# Extraction settings
MAX_RETRIES = 3
EXTRACTION_TIMEOUT = 120  # seconds

# Evaluation settings
EVALUATION_QUESTIONS_PER_DRUG = 5
MIN_CONFIDENCE_THRESHOLD = 0.7
