# app/adk/orchestrator.py - COMPLETE FIXED VERSION

# CRITICAL: Warning suppression MUST be at the very top, before any other imports
import os
import warnings
import logging

# Environment variables for GRPC and logging
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging to suppress lower-level messages
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('google.auth').setLevel(logging.ERROR)
logging.getLogger('google.cloud').setLevel(logging.ERROR)
logging.getLogger('google.generativeai').setLevel(logging.ERROR)
logging.getLogger('vertexai').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

# Custom warning filter for Gemini-specific warnings
class GeminiWarningFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage() if hasattr(record, 'getMessage') else str(record)
        warning_patterns = [
            'Warning: there are non-text parts in the response',
            'non-text parts in the response',
            'returning concatenated text result from text parts',
            'Check the full candidates.content.parts accessor'
        ]
        return not any(pattern in message for pattern in warning_patterns)

# Apply filters
for logger_name in ['google', 'google.generativeai', 'vertexai', 'grpc', 'google.cloud']:
    logger = logging.getLogger(logger_name)
    logger.addFilter(GeminiWarningFilter())
    logger.setLevel(logging.ERROR)

# NOW import the rest normally
from typing import Dict, Any, List
import json
import asyncio
import re
import sys
from io import StringIO
from google.adk.agents import Agent
from app.config.adk_config import ADK_CONFIG
from app.adk.agents import (
    create_anomaly_detector, create_summarizer, create_diversity_analyzer, create_breaking_news_alert, create_bias_detector, create_news_qa_agent, create_sentiment_agent, create_hybrid_rag_summarizer
)
from app.adk.adk_agents import (
    create_adk_anomaly_agent, create_adk_summarizer_agent, create_adk_diversity_agent, create_adk_breaking_agent, create_adk_bias_agent, create_adk_sentiment_agent, create_adk_qa_agent
)

try:
    from google.adk.runners import Runner
    from google.genai import types
except Exception:
    Runner = None
    types = None

class Orchestrator:
    def __init__(self):
        self.use_adk = ADK_CONFIG.get("adk_mode", False) and Runner is not None and types is not None
        if self.use_adk:
            self._init_adk()
        else:
            self._init_lightweight()

    def _init_lightweight(self):
        self.anomaly_detector = create_anomaly_detector()
        self.summarizer = create_summarizer()
        self.diversity_analyzer = create_diversity_analyzer()
        self.breaking_news_alert = create_breaking_news_alert()
        self.bias_detector = create_bias_detector()
        self.news_qa_agent = create_news_qa_agent()
        self.sentiment_agent = create_sentiment_agent()
        self.enable_hybrid_rag = ADK_CONFIG.get("enable_hybrid_rag_summarizer", False)
        if self.enable_hybrid_rag:
            self.hybrid_rag_summarizer = create_hybrid_rag_summarizer()

    def _init_adk(self):
        self.adk = {
            'anomaly': create_adk_anomaly_agent(),
            'summarizer': create_adk_summarizer_agent(),
            'diversity': create_adk_diversity_agent(),
            'breaking': create_adk_breaking_agent(),
            'bias': create_adk_bias_agent(),
            'sentiment': create_adk_sentiment_agent(),
            'qa': create_adk_qa_agent(),
        }

    async def process_news_workflow(self, market_data, news_articles, question=None):
        if self.use_adk:
            return await self._process_with_adk(market_data, news_articles, question)
        return await self._process_lightweight(market_data, news_articles, question)

    async def _process_lightweight(self, market_data, news_articles, question=None):
        results = {}
        results['anomalies'] = self.anomaly_detector.detect(market_data)
        results['summaries'] = self.summarizer.summarize(news_articles)
        results['diversity'] = self.diversity_analyzer.analyze(news_articles)
        results['breaking_alerts'] = self.breaking_news_alert.alert(news_articles)
        results['bias'] = self.bias_detector.detect(news_articles)
        results['sentiment'] = self.sentiment_agent.analyze(news_articles)
        if question:
            results['qa'] = self.news_qa_agent.answer(news_articles, question)
        if getattr(self, 'enable_hybrid_rag', False):
            results['rag_summary'] = self.hybrid_rag_summarizer.summarize(market_data, news_articles, question)
        return results

    async def _process_with_adk(self, market_data, news_articles, question=None):
        def text_from_news(items):
            return "\n\n".join([f"Title: {a.get('title','')}\n{a.get('content','')}" for a in items])

        app_name = "letsstock-with-ai"
        user_id = "user"
        results = {}

        async def run(agent_key, text):
            agent = self.adk.get(agent_key)
            if agent is None:
                return ""
            runner = Runner(agent=agent, app_name=app_name)
            msg = types.Content(role='user', parts=[types.Part(text=text)])
            out = ""
            async for e in runner.run_async(user_id=user_id, session_id=f"sess_{agent_key}", new_message=msg):
                if getattr(e, 'content', None) and getattr(e.content, 'parts', None):
                    for p in e.content.parts:
                        if getattr(p, 'text', None):
                            out += p.text + "\n"
            return out.strip()

        news_text = text_from_news(news_articles)
        anomalies_text = await run('anomaly', f"Market changes: {market_data}")
        summaries_text = await run('summarizer', news_text)
        diversity_text = await run('diversity', news_text)
        breaking_text = await run('breaking', news_text)
        bias_text = await run('bias', news_text)
        sentiment_text = await run('sentiment', news_text)
        qa_text = await run('qa', f"Question: {question}\n\nContext:\n{news_text}") if question else None

        results['anomalies'] = [{'text': anomalies_text}] if anomalies_text else []
        results['summaries'] = [{'text': summaries_text}] if summaries_text else []
        results['diversity'] = {'text': diversity_text} if diversity_text else {}
        results['breaking_alerts'] = [{'text': breaking_text}] if breaking_text else []
        results['bias'] = [{'text': bias_text}] if bias_text else []
        results['sentiment'] = [{'text': sentiment_text}] if sentiment_text else []
        if qa_text:
            results['qa'] = {'answer': qa_text}
        return results

    def run_all_advanced(self, market_data, news_articles, question=None):
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.process_news_workflow(market_data, news_articles, question))

# Global orchestrator instance
try:
    orchestrator = Orchestrator()
    print("🚀 TradeSage ADK Orchestrator (Clean Output Version) ready")
except Exception as e:
    print(f"❌ Failed to initialize TradeSage Orchestrator: {str(e)}")
    orchestrator = None
