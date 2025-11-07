# app/config/adk_config.py
import os
from typing import Dict, Any

# ADK Configuration
ADK_CONFIG = {
    "project_id": os.getenv("PROJECT_ID", "letsstock-with-ai"),
    "location": os.getenv("REGION", "us-central1"),
    "model": "gemini-2.0-flash",
    "use_vertex_ai": True,
    "adk_mode": os.getenv("ADK_MODE", "0") == "1",
    # Optional hybrid RAG summarizer stage toggle
    "enable_hybrid_rag_summarizer": os.getenv("ENABLE_HYBRID_RAG", "1") == "1",
}

# Agent Configuration
AGENT_CONFIGS = {
    # Old agents removed for clarity
    "anomaly_detector": {
        "name": "anomaly_detector",
        "description": "Detects unusual patterns in market data.",
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
    },
    "summarizer": {
        "name": "news_summarizer",
        "description": "Summarizes news articles.",
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
    },
    "diversity_analyzer": {
        "name": "diversity_analyzer",
        "description": "Analyzes diversity of news sources.",
        "model": "gemini-2.0-flash",
        "temperature": 0.1,
    },
    "breaking_news_alert": {
        "name": "breaking_news_alert",
        "description": "Flags breaking news articles.",
        "model": "gemini-2.0-flash",
        "temperature": 0.1,
    },
    "bias_detector": {
        "name": "bias_detector",
        "description": "Detects potential bias in news coverage.",
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
    },
    "news_qa_agent": {
        "name": "news_qa_agent",
        "description": "Answers questions from the news corpus.",
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
    },
    "sentiment_agent": {
        "name": "sentiment_agent",
        "description": "Classifies article sentiment.",
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
    },
}
