from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

MODEL_CONFIGS = {
    "qwen2": {
        "model_name": "Qwen2.5-72B-Instruct",
        "api_base": os.getenv("QWEN2_API_BASE_URL"),
    },
    "minicpm": {
        "model_name": "openbmb/MiniCPM-V-2_6",
        "api_base": os.getenv("MINICPM_API_BASE_URL"),
    },
    "phi35": {
        "model_name": "Phi-3.5-vision-instruct",
        "api_base": os.getenv("PHI35_API_BASE_URL"),
    },
    "mllama": {
        "model_name": "Llama-3.2-11B-Vision-Instruct",
        "api_base": os.getenv("MLAMA_API_BASE_URL"),
    },
    "pixtral": {
        "model_name": "Pixtral-12B-2409",
        "api_base": os.getenv("PIXTRAL_API_BASE_URL"),
    },
    "gpt4v": {
        "model_name": "gpt-4o-2024-11-20",
        "api_base": os.getenv("GPT4V_API_BASE_URL"),
    },
    "gpt4o": {
        "model_name": "gpt-4o-2024-11-20",
        "api_base": os.getenv("GPT4O_API_BASE_URL"),
    },
    "gpt-o3-mini": {
        "model_name": "o3-mini",
        "api_base": os.getenv("GPT_O3_MINI_API_BASE_URL"),
    },
    "dsv3": {
        "model_name": "deepseek-chat",
        "api_base": os.getenv("DSV3_API_BASE_URL", "https://api.deepseek.com/v1"),
    },
    "flash2v": {
        "model_name": "gemini-2.0-flash",
        "api_base": os.getenv(
            "FLASH2V_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
    },
    "flash2": {
        "model_name": "gemini-2.0-flash",
        "api_base": os.getenv(
            "FLASH2_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
    },
    "claude35": {
        "model_name": "claude-3-5-sonnet-20241022",
        "api_base": os.getenv("CLAUDE35_API_BASE_URL", "https://api.anthropic.com/v1"),
    },
    "claude37": {
        "model_name": "claude-3-7-sonnet-20250219",
        "api_base": os.getenv("CLAUDE37_API_BASE_URL", "https://api.anthropic.com/v1"),
    },
    "mistral-large": {
        "model_name": "mistral-large-latest",
        "api_base": os.getenv(
            "MISTRAL_LARGE_API_BASE_URL",
            "https://api.mistral.ai/v1",
        ),
    },
}
