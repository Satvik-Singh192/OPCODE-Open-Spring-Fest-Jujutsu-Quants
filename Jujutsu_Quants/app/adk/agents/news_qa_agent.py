# Minor doc update: re-submitting for label and tracking

import re
import math
from collections import Counter
from typing import List, Dict, Any, TypedDict, Union

from app.config.adk_config import AGENT_CONFIGS

# --- Type Definitions for API Contract ---

class Passage(TypedDict):
    text: str
    source: str
    start: int
    end: int

class Citation(TypedDict):
    source: str
    start: int
    end: int

class RankedPassage(TypedDict):
    passage: Passage
    score: float

class Answer(TypedDict):
    answer: str
    citations: List[Citation]


# --- Default Constants ---

DEFAULT_CHUNK_SIZE = 150
DEFAULT_CHUNK_OVERLAP = 30
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.05


# --- Agent Instruction ---

QA_INSTRUCTION = """
You are the News QA Agent. Answer user questions using the provided news corpus.
Be concise and cite the article titles or URLs.
If no relevant answer is found, reply with 'No relevant article found.'
"""


# --- Factory Function ---

def create_news_qa_agent():
    config = AGENT_CONFIGS["news_qa_agent"]

    class NewsQAAgent:
        def _init_(self):
            self.name = config["name"]
            self.model = config["model"]
            self.description = config["description"]
            self.instruction = QA_INSTRUCTION
            self.tools = []

        # ------------------------------------------------------------
        # 1️⃣ Passage-based extractive QA (Your version)
        # ------------------------------------------------------------
        def answer(
            self,
            articles: List[Dict[str, Any]],
            question: str,
            top_k: int = DEFAULT_TOP_K,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            overlap: int = DEFAULT_CHUNK_OVERLAP,
            min_score: float = DEFAULT_MIN_SCORE
        ) -> Answer:
            """Returns an extractive answer with citations from a list of articles."""
            fallback_answer = "No relevant article found."
            passages: List[Passage] = []
            top_passages: List[RankedPassage] = []

            if articles:
                passages = _chunk_articles(articles, chunk_size, overlap)

            if passages:
                ranked_passages = _rank_passages(passages, question)
                top_passages = [r for r in ranked_passages[:top_k] if r['score'] > min_score]

            if not top_passages:
                return {"answer": fallback_answer, "citations": []}

            final_answer = " ".join([r['passage']['text'] for r in top_passages])
            final_citations: List[Citation] = [
                {"source": r['passage']['source'], "start": r['passage']['start'], "end": r['passage']['end']}
                for r in top_passages
            ]

            return {"answer": final_answer, "citations": final_citations}

        # ------------------------------------------------------------
        # 2️⃣ Relevance scoring (from main)
        # ------------------------------------------------------------
        def _calculate_relevance_score(self, article, question):
            content = article.get("content", "").lower()
            title = article.get("title", "").lower()
            question_lower = question.lower()

            score = 0
            question_words = set(question_lower.split())
            content_words = set(content.split())
            title_words = set(title.split())

            # Weighted keyword overlap
            title_matches = len(question_words.intersection(title_words))
            score += title_matches * 3

            content_matches = len(question_words.intersection(content_words))
            score += content_matches

            for word in question_words:
                if word in content:
                    score += 0.5

            return score

        def _rank_articles_by_relevance(self, articles, question):
            scored_articles = []
            for article in articles:
                score = self._calculate_relevance_score(article, question)
                if score > 0:
                    scored_articles.append((article, score))

            scored_articles.sort(key=lambda x: x[1], reverse=True)
            return [article for article, score in scored_articles]

        # ------------------------------------------------------------
        # 3️⃣ Question-type detection (from main)
        # ------------------------------------------------------------
        def _detect_question_type(self, question):
            q = question.lower()
            temporal_keywords = ["trend", "change", "over time", "recently", "history", "evolution", "growth", "decline", "past", "has the", "has been"]
            if any(k in q for k in temporal_keywords):
                return "temporal"
            if any(k in q for k in ["compare", "difference", "vs", "versus"]):
                return "comparative"
            if any(k in q for k in ["impact", "effect", "consequence", "cause"]):
                return "causal"
            if any(k in q for k in ["what", "how", "why", "when", "where"]):
                return "factual"
            return "general"

        # ------------------------------------------------------------
        # 4️⃣ Combined answer logic
        # ------------------------------------------------------------
        def _extract_relevant_excerpts(self, article, question, max_length=200):
            content = article.get("content", "")
            question_words = set(question.lower().split())
            sentences = content.split(". ")
            scored_sentences = [(s, sum(1 for w in question_words if w in s.lower())) for s in sentences if sum(1 for w in question_words if w in s.lower()) > 0]
            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            excerpt = ""
            for sentence, _ in scored_sentences:
                if len(excerpt + sentence) < max_length:
                    excerpt += sentence + ". "
                else:
                    break
            return excerpt.strip()

    return NewsQAAgent()


# --- Helper Functions ---

def _chunk_articles(articles: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Passage]:
    passages: List[Passage] = []
    for article in articles:
        content = article.get('content', '')
        if not content.strip():
            continue
        source = article.get('source_url', article.get('title', 'unknown_source'))
        words = list(re.finditer(r'\S+', content))
        if not words:
            continue
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            if start_idx >= end_idx:
                break
            start_char = words[start_idx].start()
            end_char = words[end_idx - 1].end()
            passages.append({"text": content[start_char:end_char], "source": source, "start": start_char, "end": end_char})
            next_start_idx = start_idx + chunk_size - overlap
            start_idx = max(next_start_idx, start_idx + 1)
    return passages


def _rank_passages(passages: List[Passage], question: str) -> List[RankedPassage]:
    passage_texts = [p['text'] for p in passages]
    passage_tokens = [_tokenize(t) for t in passage_texts]
    question_tokens = _tokenize(question)
    corpus_tokens = passage_tokens + [question_tokens]
    idf_scores = _compute_idf(corpus_tokens)
    vocab = sorted(idf_scores.keys())
    question_vector = _compute_tfidf_vector(question_tokens, idf_scores, vocab)
    ranked: List[RankedPassage] = []
    for p_tokens, p in zip(passage_tokens, passages):
        passage_vector = _compute_tfidf_vector(p_tokens, idf_scores, vocab)
        score = _cosine_similarity(question_vector, passage_vector)
        ranked.append({"passage": p, "score": score})
    return sorted(ranked, key=lambda x: x['score'], reverse=True)


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {w: c / total for w, c in counts.items()}


def _compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    num_docs = len(documents)
    if num_docs == 0:
        return {}
    df = Counter()
    for doc in documents:
        df.update(set(doc))
    return {w: math.log(num_docs / (1 + count)) + 1 for w, count in df.items()}


def _compute_tfidf_vector(tokens: List[str], idf_scores: Dict[str, float], vocab: List[str]) -> List[float]:
    tf = _compute_tf(tokens)
    return [tf.get(word, 0) * idf_scores.get(word, 0) for word in vocab]


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a*b for a, b in zip(vec1, vec2))
    mag1 = math.hypot(*vec1)
    mag2 = math.hypot(*vec2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)