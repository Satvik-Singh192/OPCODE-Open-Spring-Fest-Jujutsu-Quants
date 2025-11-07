from typing import List, Dict, Any, Tuple

from app.config.adk_config import AGENT_CONFIGS
from .bias_detector import create_bias_detector


def _classify_regime(change_fraction: float) -> str:
    """Classify price move regime from fractional change. Deterministic thresholds.
    flat: |Δ| < 0.002, up: Δ >= 0.002, down: Δ <= -0.002
    """
    if change_fraction is None:
        return "flat"
    if change_fraction >= 0.002:
        return "up"
    if change_fraction <= -0.002:
        return "down"
    return "flat"


def _chunk_passages(text: str, window: int = 220, stride: int = 180) -> List[Tuple[int, int, str]]:
    """Create overlapping fixed-size character windows to keep implementation tiny and deterministic.
    Returns list of (start, end, passage_text).
    """
    if not text:
        return []
    passages: List[Tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + window, n)
        passages.append((i, end, text[i:end]))
        if end == n:
            break
        i += stride
    return passages


def _score_passage(passage: str, query_terms: List[str]) -> int:
    passage_lower = passage.lower()
    score = 0
    for t in query_terms:
        if t and t in passage_lower:
            score += 1
    return score


def create_hybrid_rag_summarizer():
    config = AGENT_CONFIGS.get("summarizer", {"name": "hybrid_rag_summarizer"})

    class HybridRAGSummarizer:
        def __init__(self):
            self.name = config.get("name", "hybrid_rag_summarizer")
            self.bias = create_bias_detector()

        def _compute_regimes(self, market_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            regimes = []
            for item in market_data or []:
                sym = item.get("symbol")
                chg = item.get("price_change")
                regimes.append({
                    "symbol": sym,
                    "price_change": chg,
                    "regime": _classify_regime(chg if isinstance(chg, (int, float)) else 0.0),
                })
            return regimes

        def _rank_passages(self, articles: List[Dict[str, Any]], question: str = None) -> List[Dict[str, Any]]:
            query_terms: List[str] = []
            if question:
                query_terms = [w for w in question.lower().split() if len(w) > 2]
            else:
                # fallback to frequent title tokens as naive signal
                tokens: List[str] = []
                for a in articles or []:
                    tokens.extend([w.lower() for w in (a.get("title") or "").split() if len(w) > 3])
                # keep a small set to avoid heavy logic
                seen = set()
                for t in tokens:
                    if t not in seen:
                        query_terms.append(t)
                        seen.add(t)
                        if len(query_terms) >= 10:
                            break

            ranked: List[Dict[str, Any]] = []
            for art in articles or []:
                content = art.get("content", "")
                url = art.get("url") or art.get("source") or ""
                for start, end, chunk in _chunk_passages(content):
                    score = _score_passage(chunk, query_terms)
                    if score > 0:
                        ranked.append({
                            "source": url,
                            "start": start,
                            "end": end,
                            "text": chunk,
                            "score": score,
                        })
            ranked.sort(key=lambda x: x["score"], reverse=True)
            return ranked[:3]

        def summarize(self, market_data: List[Dict[str, Any]], news_articles: List[Dict[str, Any]], question: str = None) -> Dict[str, Any]:
            regimes = self._compute_regimes(market_data)
            top_passages = self._rank_passages(news_articles, question)
            bias_flags = self.bias.detect(news_articles)

            # Build key points from top passages' first clause and titles
            key_points: List[str] = []
            for p in top_passages:
                snippet = p["text"].strip().split(". ")[0].strip()
                if snippet:
                    key_points.append(snippet[:120])
            if not key_points:
                # fallback: use titles
                for a in news_articles or []:
                    t = (a.get("title") or "").strip()
                    if t:
                        key_points.append(t)
                        if len(key_points) >= 3:
                            break

            # Compact regime sentence
            moves = []
            for r in regimes:
                sym = r.get("symbol")
                reg = r.get("regime")
                if sym:
                    moves.append(f"{sym}:{reg}")
            move_sentence = ", ".join(moves) if moves else "market flat"

            # Uncertainty factors from bias reasons/entities
            uncertainty: List[str] = []
            for b in bias_flags:
                reason = b.get("reason")
                ent = b.get("entity_focus")
                if reason:
                    uncertainty.append(reason)
                elif ent:
                    uncertainty.append(f"bias around {ent}")
                if len(uncertainty) >= 4:
                    break

            # Citations from passages
            citations = [{"source": p.get("source", ""), "start": p["start"], "end": p["end"]} for p in top_passages]

            # Compose brief summary
            headline = "; ".join(key_points[:2]) if key_points else "Key drivers mixed"
            summary = f"{headline}. Moves: {move_sentence}."

            return {
                "summary": summary,
                "key_points": key_points[:5],
                "uncertainty_factors": uncertainty,
                "citations": citations,
            }

    return HybridRAGSummarizer()


