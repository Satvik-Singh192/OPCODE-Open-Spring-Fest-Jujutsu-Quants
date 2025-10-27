# from app.config.adk_config import AGENT_CONFIGS

# QA_INSTRUCTION = """
# You are the News QA Agent. Answer user questions using the news corpus. Be concise and cite the article title in your answer. If no answer is found, say 'No relevant article found.'
# """

# def create_news_qa_agent():
#     config = AGENT_CONFIGS["news_qa_agent"]
#     class NewsQAAgent:
#         def __init__(self):
#             self.name = config["name"]
#             self.model = config["model"]
#             self.description = config["description"]
#             self.instruction = QA_INSTRUCTION
#             self.tools = []
#         def answer(self, articles, question):
#             keywords = question.lower().split() if question else []
#             for article in articles:
#                 content = article.get('content', '').lower()
#                 if any(word in content for word in keywords):
#                     return {
#                         'title': article.get('title', ''),
#                         'answer': article.get('content', '')
#                     }
#             return {'answer': 'No relevant article found.'}
#     return NewsQAAgent()

# app/adk/agents/news_qa_agent.py

from app.config.adk_config import AGENT_CONFIGS

QA_INSTRUCTION = """
You are the News QA Agent. Answer user questions using the provided news corpus.
Be concise and cite the article titles in your answer.
If no relevant answer is found, reply with 'No relevant article found.'
"""

def create_news_qa_agent():
    config = AGENT_CONFIGS["news_qa_agent"]

    class NewsQAAgent:
        def __init__(self):
            self.name = config["name"]
            self.model = config["model"]
            self.description = config["description"]
            self.instruction = QA_INSTRUCTION
            self.tools = []

        # ------------------------------------------------------------
        # 1️⃣ Relevance Scoring
        # ------------------------------------------------------------
        def _calculate_relevance_score(self, article, question):
            """Calculate a simple relevance score between article and question."""
            content = article.get("content", "").lower()
            title = article.get("title", "").lower()
            question_lower = question.lower()

            score = 0
            question_words = set(question_lower.split())
            content_words = set(content.split())
            title_words = set(title.split())

            # Weighted keyword overlap
            title_matches = len(question_words.intersection(title_words))
            score += title_matches * 3  # Title matches are more important

            content_matches = len(question_words.intersection(content_words))
            score += content_matches * 1

            # Phrase (word presence) bonus
            for word in question_words:
                if word in content:
                    score += 0.5

            return score

        def _rank_articles_by_relevance(self, articles, question):
            """Rank articles by calculated relevance score."""
            scored_articles = []
            for article in articles:
                score = self._calculate_relevance_score(article, question)
                if score > 0:
                    scored_articles.append((article, score))

            scored_articles.sort(key=lambda x: x[1], reverse=True)
            return [article for article, score in scored_articles]

        # ------------------------------------------------------------
        # 2️⃣ Answer Extraction
        # ------------------------------------------------------------
        def _extract_relevant_excerpts(self, article, question, max_length=200):
            """Extract the most relevant excerpts from article content."""
            content = article.get("content", "")
            question_words = set(question.lower().split())

            # Split into sentences
            sentences = content.split(". ")

            scored_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                matches = sum(1 for word in question_words if word in sentence_lower)
                if matches > 0:
                    scored_sentences.append((sentence, matches))

            # Sort by relevance
            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            excerpt = ""
            for sentence, _ in scored_sentences:
                if len(excerpt + sentence) < max_length:
                    excerpt += sentence + ". "
                else:
                    break

            return excerpt.strip()

        # ------------------------------------------------------------
        # 3️⃣ Question Type Detection
        # ------------------------------------------------------------
        def _detect_question_type(self, question):
            """Detect the type of question being asked."""
            question_lower = question.lower()

            # Temporal questions FIRST
            temporal_keywords = ["trend", "change", "over time", "recently", "history", "evolution", "growth", "decline", "past", "has the", "has been"]
            if any(word in question_lower for word in temporal_keywords):
                return "temporal"

             # Comparative questions
            if any(word in question_lower for word in ["compare", "difference", "vs", "versus"]):
                return "comparative"

            # Causal questions
            if any(word in question_lower for word in ["impact", "effect", "consequence", "cause"]):
                return "causal"

            # Factual/general questions LAST
            if any(word in question_lower for word in ["what", "how", "why", "when", "where"]):
                return "factual"

            return "general"


        # ------------------------------------------------------------
        # 4️⃣ Handling Question Types
        # ------------------------------------------------------------
        def _handle_question_type(self, question, articles):
            """Route handling logic based on question type."""
            question_type = self._detect_question_type(question)

            if question_type == "comparative":
                return self._handle_comparative_question(question, articles)
            elif question_type == "temporal":
                return self._handle_temporal_question(question, articles)
            elif question_type == "causal":
                return self._handle_causal_question(question, articles)
            else:
                return self._handle_general_question(question, articles)

        def _handle_general_question(self, question, articles):
            """Default handling for factual/general questions."""
            return self._generate_answer(articles, question)

        def _handle_comparative_question(self, question, articles):
            """Compare information across sources."""
            return self._generate_answer(articles, question)

        def _handle_temporal_question(self, question, articles):
            """Handle time/trend-based questions."""
            return self._generate_answer(articles, question)

        def _handle_causal_question(self, question, articles):
            """Handle cause-effect questions."""
            return self._generate_answer(articles, question)

        # ------------------------------------------------------------
        # 5️⃣ Generate Final Answer
        # ------------------------------------------------------------
        def _generate_answer(self, articles, question):
            """Generate a grounded answer using multiple top articles."""
            if not articles:
                return "No relevant articles found."

            top_articles = articles[:3]
            answers = []

            for i, article in enumerate(top_articles):
                excerpt = self._extract_relevant_excerpts(article, question)
                if excerpt:
                    source = f"Source {i+1}: {article.get('title', 'Untitled')}"
                    answers.append(f"{source}\n{excerpt}")

            if not answers:
                return "No relevant information found in the articles."

            return "\n\n".join(answers)

        # ------------------------------------------------------------
        # 6️⃣ Enhanced Public API
        # ------------------------------------------------------------
        def answer(self, articles, question):
            """Enhanced answer method with scoring, extraction, and citations."""
            if not articles or not question:
                return {"answer": "No articles or question provided."}

            relevant_articles = self._rank_articles_by_relevance(articles, question)

            if not relevant_articles:
                return {"answer": "No relevant articles found for this question."}

            answer_text = self._handle_question_type(question, relevant_articles)
            sources = [article.get("title", "Untitled") for article in relevant_articles[:3]]

            return {
                "answer": answer_text,
                "sources": sources,
                "relevance_score": self._calculate_relevance_score(relevant_articles[0], question),
                "question_type": self._detect_question_type(question),
            }

    return NewsQAAgent()



agent = create_news_qa_agent()
articles = [
    {
        "title": "Apple Reports Strong Earnings",
        "content": "Apple Inc. reported strong quarterly earnings, beating analyst expectations."
    },
    {
        "title": "Market Update",
        "content": "The stock market showed mixed signals today."
    }
]
result = agent.answer(articles, "What did Apple report about earnings?")
print(result)