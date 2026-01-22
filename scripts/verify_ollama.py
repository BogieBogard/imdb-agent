
try:
    from agent import MovieAgent
    print("Agent imported successfully.")
    
    # Initialize with dummy key
    agent = MovieAgent(api_key="ollama_local")
    print("Agent initialized successfully.")
    
    # Test simple query that requires pandas agent
    response = agent.run("How many movies are in the dataset?")
    print(f"Response: {response}")
    
    # Test vector search (will trigger index creation if not exists)
    response_search = agent.run("Find me a movie about a space odyssey.")
    print(f"Search Response: {response_search}")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
