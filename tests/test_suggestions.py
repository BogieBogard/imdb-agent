import pytest

def test_suggestions_known_movie(movie_agent):
    """
    Verifies that the agent generates valid suggestions for a known movie.
    Ensures the original movie title is not suggested back to the user.
    """
    response_text = "The Godfather is a classic crime drama."
    suggestions = movie_agent.get_suggestions(response_text)
    
    # Check that we have suggestions
    assert suggestions is not None, "Failed to generate suggestions for a known movie."
    
    # Check format
    assert "### 🎥 You might also like:" in suggestions
    
    # Check that the EXACT original title is not suggested (looking for bold marker)
    # The agent suggests "**Movie Title**", so checking for "**The Godfather**" ensures
    # we don't flag "The Godfather: Part II" as a duplicate of "The Godfather".
    assert "**The Godfather**" not in suggestions, "Original title was suggested back to the user."
    
    # Basic check for expected similar movies (determinism isn't guaranteed with LLM extraction, 
    # but the similarity search is deterministic if the extraction works)
    # Just checking for length to ensure we got some results.
    assert len(suggestions.split("\n")) > 1

def test_suggestions_no_movie(movie_agent):
    """
    Verifies that the agent returns None when no movies are identified in the text.
    """
    response_text = "I don't know any movies about that topic."
    suggestions = movie_agent.get_suggestions(response_text)
    
    assert suggestions is None, "Should not generate suggestions for text without movies."
