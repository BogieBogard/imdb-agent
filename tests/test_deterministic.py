import pytest
from tests.utils import llm_evaluate

def test_question_1_matrix_release(movie_agent):
    """
    Question: When did the Matrix release?
    Answer: The Matrix was released in 1999.
    """
    response = movie_agent.run("When did the Matrix release?")
    assert "1999" in response
    assert "Matrix" in response

def test_question_3_comedy_movies_2010_2020(movie_agent):
    """
    Question: Top 7 comedy movies between 2010-2020 by IMDB rating?
    """
    response = movie_agent.run("Top 7 comedy movies between 2010-2020 by IMDB rating?")
    
    expected_movies = [
        "Gisaengchung",
        "The Intouchables",
        "Chhichhore",
        "Green Book",
        "Three Billboards Outside Ebbing, Missouri",
        "Klaus",
        "Queen"
    ]
    
    # Check that at least most expected movies are present
    found_count = sum(1 for movie in expected_movies if movie in response)
    assert found_count >= 5, f"Expected at least 5 of {expected_movies} to be in response. Got: {response}"

def test_question_5_top_directors_gross(movie_agent):
    """
    Question: Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice.
    """
    response = movie_agent.run("Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice.")
    
    assert "Anthony Russo" in response
    assert "James Cameron" in response
    assert "Avengers: Endgame" in response
    assert "Avatar" in response



def test_question_6_top_10_movies_1m_votes_low_gross_deterministic(movie_agent):
    """
    Question: Top 10 movies with over 1M votes but lower gross earnings.
    Deterministic Check:
    - Failing condition: The response contains a movie that is NOT in the allowed inline list.
    """
    question = "Top 10 movies with over 1M votes but lower gross earnings."
    response = movie_agent.run(question)
    
    # Allowed movies (Movies with > 1M votes)
    allowed_movies = [
        "The Godfather", "The Lord of the Rings: The Return of the King", "Pulp Fiction", "Schindler's List",
        "The Lord of the Rings: The Fellowship of the Ring", "Saving Private Ryan", "Goodfellas", "Star Wars",
        "The Godfather: Part II", "The Lord of the Rings: The Two Towers", "Back to the Future", "The Departed",
        "The Silence of the Lambs", "The Dark Knight", "American Beauty", "Avatar", "Forrest Gump",
        "Star Wars: Episode V - The Empire Strikes Back", "Django Unchained", "Memento", "The Shawshank Redemption",
        "The Dark Knight Rises", "Guardians of the Galaxy", "The Wolf of Wall Street", "Titanic", "Interstellar",
        "Inception", "The Matrix", "Batman Begins", "The Avengers", "Inglourious Basterds", "Kill Bill: Vol. 1",
        "Gladiator", "The Prestige", "Fight Club", "Se7en", "Léon", "Shutter Island",
        "Pirates of the Caribbean: The Curse of the Black Pearl", "V for Vendetta", "American History X", "The Green Mile"
    ]
    
    # Load Main Dataset to identify "Movies that exist but are invalid answers"
    import pandas as pd
    df = pd.read_csv("imdb_top_1000.csv")
    all_titles = set(df['Series_Title'].unique())
    allowed_titles = set(allowed_movies)
    
    # Identify forbidden movies (Movies in DB but < 1M votes)
    forbidden_titles = all_titles - allowed_titles
    
    # Check if any forbidden title is in the response
    found_forbidden = []
    for title in forbidden_titles:
        # The agent output format is consistently: **Title (Year)**
        # We search for the Bolded Title pattern to avoid substrings like 'Ray' matching 'Ray Liotta'
        if f"**{title}" in response:
            found_forbidden.append(title)
            
    assert not found_forbidden, f"Response contains movies with < 1M votes: {found_forbidden}. Response: {response}"
    
    # Pass if no forbidden movies found

def test_question_4_horror_movies_meta_85_imdb_8(movie_agent):
    """
    Question: Top horror movies with a meta score above 85 and IMDB rating above 8
    
    Deterministic Check:
    - Must contain "Psycho" and "Alien".
    - Should NOT contain "The Shining" (Meta < 85) or "The Thing" (Meta < 85).
    """
    response = movie_agent.run("Top horror movies with a meta score above 85 and IMDB rating above 8")
    
    # Assertions for correct movies
    assert "Psycho" in response
    assert "Alien" in response
    
    # Assertions for exclusion of close-but-fail cases
    assert "The Shining" not in response
    assert "The Thing" not in response
