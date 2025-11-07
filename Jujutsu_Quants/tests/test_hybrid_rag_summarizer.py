import pytest

from app.adk.agents.hybrid_rag_summarizer import create_hybrid_rag_summarizer, _classify_regime


def test_regime_classification():
    assert _classify_regime(0.01) == "up"
    assert _classify_regime(-0.02) == "down"
    assert _classify_regime(0.0005) == "flat"
    assert _classify_regime(None) == "flat"


def test_passage_fusion_and_citations():
    agent = create_hybrid_rag_summarizer()
    market = [{"symbol": "AAPL", "price_change": 0.012}]
    articles = [
        {
            "title": "Apple iPhone demand improves as supply chain eases",
            "url": "https://example.com/news1",
            "content": (
                "Apple reported better iPhone demand this quarter. "
                "Analysts noted supply chain easing in China factories. "
                "Investors reacted positively to the update."
            ),
        }
    ]
    out = agent.summarize(market, articles, question="What are key drivers today?")
    assert isinstance(out, dict)
    assert "summary" in out and isinstance(out["summary"], str)
    assert "key_points" in out and len(out["key_points"]) >= 1
    assert "uncertainty_factors" in out
    assert "citations" in out and isinstance(out["citations"], list)
    if out["citations"]:
        c = out["citations"][0]
        assert set(c.keys()) == {"source", "start", "end"}

