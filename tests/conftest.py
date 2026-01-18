import pytest
import os
from agent import MovieAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture(scope="session")
def movie_agent():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment variables")
    return MovieAgent(api_key)
