import pytest
from tests.utils import llm_evaluate

def test_question_2_top_5_movies_2019(movie_agent):
    """
    Question: What are the top 5 movies of 2019 by meta score?
    Expected: Should provide 5 movies from 2019. Previously hallucinated 'unable to provide'.
    """
    question = "What are the top 5 movies of 2019 by meta score?"
    response = movie_agent.run(question)
    
    # We expect this to FAIL if the bug persists (agent says it can't find movies)
    # OR PASS if the agent correctly identifies 2019 movies.
    # The criteria is strictly: "Did it return a list of 5 movies released in 2019?"
    
    criteria = """
    1. The answer must contain a list of 5 movies.
    2. The movies must be from the year 2019.
    3. The answer should NOT state that the dataset does not contain movies from 2019.
    """
    
    passed, reason = llm_evaluate(question, response, criteria)
    assert passed, f"Judge failed the response. Reason: {reason}\nAgent Response: {response}"

def test_question_6_top_10_movies_1m_votes_low_gross(movie_agent):
    """
    Question: Top 10 movies with over 1M votes but lower gross earnings.
    """
    question = "Top 10 movies with over 1M votes but lower gross earnings."
    response = movie_agent.run(question)
    
    criteria = """
    1. The answer should attempt to list 10 movies (or as many as fit criteria if fewer than 10).
    2. All listed movies must have > 1,000,000 votes.
    3. "Lower gross earnings" implies lower than average or relatively low for such high votes.
    4. The list should explicitly mention vote counts and gross earnings.
    """
    
    passed, reason = llm_evaluate(question, response, criteria)
    assert passed, f"Judge failed the response. Reason: {reason}\nAgent Response: {response}"

def test_question_8_spielberg_scifi_summaries(movie_agent):
    """
    Question: Summarize the movie plots of Steven Spielberg’s top-rated sci-fi movies.
    """
    question = "Summarize the movie plots of Steven Spielberg’s top-rated sci-fi movies."
    response = movie_agent.run(question)
    
    criteria = """
    1. The answer must list Steven Spielberg sci-fi movies.
    2. It must provide plot summaries for them.
    3. CRITICAL: It MUST NOT include "War of the Worlds" (2005) if that movie is not in the source dataset (imdb_top_1000.csv).
       (Context: Previous runs hallucinated War of the Worlds. If the agent finds it legitimately, that's fine,
       but if the dataset lacks it, it shouldn't appear.)
    4. "A.I. Artificial Intelligence" is another potential hallucination to watch out for if not in data.
    """
    
    passed, reason = llm_evaluate(question, response, criteria)
    assert passed, f"Judge failed the response. Reason: {reason}\nAgent Response: {response}"

def test_question_9_police_movies_before_1990(movie_agent):
    """
    Question: List of movies before 1990 that have involvement of police in the plot.
    """
    question = "List of movies before 1990 that have involvement of police in the plot. Hint, this needs to be based on similarity search and not just the word police."
    response = movie_agent.run(question)
    
    criteria = """
    1. The listed movies must have release years strictly before 1990.
    2. The plot descriptions must involve police.
    3. The answer should not include movies from 1990 or later (e.g., L.A. Confidential - 1997, Se7en - 1995 are fail cases).
    """
    
    passed, reason = llm_evaluate(question, response, criteria)
    assert passed, f"Judge failed the response. Reason: {reason}\nAgent Response: {response}"
