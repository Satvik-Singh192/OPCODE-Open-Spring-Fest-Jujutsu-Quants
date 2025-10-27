import pytest
from app.adk.agents.news_qa_agent import create_news_qa_agent

def test_relevance_scoring():
    """Test relevance scoring functionality"""
    agent = create_news_qa_agent()

    articles = [
        {
            'title': 'Apple Reports Strong Earnings',
            'content': 'Apple Inc. reported strong quarterly earnings, beating analyst expectations.'
        },
        {
            'title': 'Market Update',
            'content': 'The stock market showed mixed signals today.'
        }
    ]

    question = "What did Apple report about earnings?"

    score1 = agent._calculate_relevance_score(articles[0], question)
    score2 = agent._calculate_relevance_score(articles[1], question)

    assert score1 > score2, "Apple article should have higher relevance"

def test_answer_extraction():
    """Test answer extraction from articles"""
    agent = create_news_qa_agent()

    article = {
        'title': 'Apple Reports Strong Earnings',
        'content': (
            'Apple Inc. reported strong quarterly earnings, beating analyst expectations. '
            'The company saw significant growth in iPhone sales. Revenue increased by 15% compared to last year.'
        )
    }

    question = "What did Apple report about earnings?"
    excerpt = agent._extract_relevant_excerpts(article, question)

    assert 'earnings' in excerpt.lower(), "Excerpt should mention earnings"
    assert len(excerpt) < 200, "Excerpt should be concise (<200 chars)"

def test_question_type_detection():
    """Test question type detection"""
    agent = create_news_qa_agent()

    assert agent._detect_question_type("What is Apple's revenue?") == 'factual'
    assert agent._detect_question_type("Compare Apple and Microsoft") == 'comparative'
    assert agent._detect_question_type("How has the market changed recently?") == 'temporal'

def test_enhanced_answer():
    """Test enhanced answer method"""
    agent = create_news_qa_agent()

    articles = [
        {
            'title': 'Apple Reports Strong Earnings',
            'content': 'Apple Inc. reported strong quarterly earnings, beating analyst expectations.'
        }
    ]

    question = "What did Apple report about earnings?"
    result = agent.answer(articles, question)

    assert 'answer' in result
    assert 'sources' in result
    assert 'relevance_score' in result
    assert 'question_type' in result
    assert 'earnings' in result['answer'].lower()
