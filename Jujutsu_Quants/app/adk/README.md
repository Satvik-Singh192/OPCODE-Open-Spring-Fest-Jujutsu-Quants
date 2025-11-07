# Core Agent Framework (OpenAI News Lab)

This directory contains the core agent framework for advanced news and market data analysis. The current agent set includes:

- AnomalyDetector: Detects unusual patterns in market data.
- NewsSummarizer: Summarizes news articles.
- DiversityAnalyzer: Analyzes diversity of news sources.
- BreakingNewsAlert: Flags breaking news articles.
- BiasDetector: Detects potential bias in news coverage.
- NewsQAAgent: Answers questions from the news corpus.

## Hybrid RAG Summarizer (optional)

- `HybridRAGSummarizer`: Fuses retrieval-ranked passages, market move regimes, and bias flags into a cited summary available under `rag_summary` in the `/api/v2/report` response.
- Enable/disable via config flag `ENABLE_HYBRID_RAG` (env var). Default: enabled.
- Minimal usage:

```bash
uvicorn Jujutsu-Quants.app.adk.main:app --reload
curl -s -X POST http://localhost:8000/api/v2/report -H "Content-Type: application/json" -d '{
  "symbols": ["AAPL"],
  "news_urls": ["https://example.com/news1"],
  "question": "What are key drivers today?"
}' | jq
```


To add a new agent, create a new Python file in `agents/` and integrate it in `orchestrator.py`.
