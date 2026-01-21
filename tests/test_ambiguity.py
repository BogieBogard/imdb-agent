import pytest
from unittest.mock import MagicMock, patch
from agent import MovieAgent

class TestAmbiguityModule:
    
    @pytest.fixture
    def agent(self):
        # Create agent with a dummy key (mocking init to avoid real network calls if possible, 
        # but MovieAgent init does a lot. For unit testing logic, we might need to be careful).
        # Since MovieAgent loads CSV and builds VectorStore, we'll try to just mock the LLM part
        # AFTER initialization if possible, or Mock the whole class and test method logic?
        # A simpler approach for unit testing the logic added to 'run' and '_check_ambiguity' 
        # without full init is to subclass or mock.
        
        # However, to keep it simple and consistent with existing tests, let's assume we can 
        # instantiate it or we mock the expensive parts.
        # Let's mock the valid init process to avoid overhead:
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-key'}):
            with patch('agent.pd.read_csv') as mock_read_csv:
                with patch('agent.OllamaEmbeddings'):
                    with patch('agent.FAISS'):
                        with patch('agent.initialize_agent'):
                            with patch('agent.create_pandas_dataframe_agent'):
                                # Return a dummy df
                                import pandas as pd
                                mock_read_csv.return_value = pd.DataFrame({
                                    'Series_Title': ['A'], 'Released_Year': [2000], 
                                    'Gross': ['100'], 'No_of_Votes': [1], 'Meta_score': [1],
                                    'Overview': ['Plot'], 'Genre': ['Action'], 'Director': ['D']
                                })
                                
                                agent = MovieAgent(api_key="fake-key")
                                # Fix the llm to be a mock so we can control it
                                agent.llm = MagicMock()
                                agent.master_agent = MagicMock()
                                return agent

    def test_low_ambiguity_flow(self, agent):
        """Test that a low ambiguity score proceeds to normal execution."""
        
        # Mock _check_ambiguity to return low score
        # We can either mock the LLM response OR mock the method itself. 
        # Mocking the method is safer for testing the 'run' flow logic.
        agent._check_ambiguity = MagicMock(return_value={"score": 10, "clarification": ""})
        
        agent.master_agent.run.return_value = "Final Answer"
        
        response = agent.run("Clear Query")
        
        # Assertions
        assert response == "Final Answer"
        agent._check_ambiguity.assert_called_with("Clear Query")
        agent.master_agent.run.assert_called_with("Clear Query")
        assert agent.pending_ambiguous_query is None

    def test_high_ambiguity_trigger(self, agent):
        """Test that high ambiguity triggers clarification and sets state."""
        
        clarification_q = "What do you mean?"
        agent._check_ambiguity = MagicMock(return_value={"score": 90, "clarification": clarification_q})
        
        response = agent.run("Ambiguous Query")
        
        # Assertions
        assert response == clarification_q
        assert agent.pending_ambiguous_query == "Ambiguous Query"
        agent.master_agent.run.assert_not_called()

    def test_clarification_resolution_flow(self, agent):
        """Test that answering the clarification resolves the pending query."""
        
        # 1. Set up state as if we just asked a clarification
        agent.pending_ambiguous_query = "Ambiguous Query"
        
        agent.master_agent.run.return_value = "Result after clarification"
        
        # 2. User provides the answer
        clarification_answer = "Clarification Answer"
        response = agent.run(clarification_answer)
        
        # Assertions
        assert response == "Result after clarification"
        assert agent.pending_ambiguous_query is None
        
        # Check that master_agent was called with combined context
        # We need to check the actual string passed
        call_args = agent.master_agent.run.call_args[0][0]
        assert "Ambiguous Query" in call_args
        assert "Clarification Answer" in call_args

    def test_check_ambiguity_payload_parsing(self, agent):
        """Test the parsing logic inside _check_ambiguity using a mocked LLM response."""
        
        # We need to un-mock _check_ambiguity if we mocked it in previous tests, 
        # but since 'agent' is a fixture, it's fresh per test.
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = '```json\n{"score": 60, "clarification": "Context?"}\n```'
        agent.llm.invoke.return_value = mock_response
        
        result = agent._check_ambiguity("Test Query")
        
        assert result["score"] == 60
        assert result["clarification"] == "Context?"
