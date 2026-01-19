from openai import OpenAI
import os

def llm_evaluate(question, actual_answer, expected_criteria):
    """
    Evaluates the agent's answer using an LLM judge.
    Returns (bool, str): (passed, reason)
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
        
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    You are an impartial judge evaluating the performance of a movie question-answering bot.
    
    Question: {question}
    
    Actual Answer: {actual_answer}
    
    Evaluation Criteria: 
    {expected_criteria}
    
    Does the Actual Answer meet the evaluation criteria? 
    Respones with specific "YES" or "NO" on the first line, followed by a brief explanation.
    """
    
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    content = response.choices[0].message.content.strip()
    passed = content.upper().startswith("YES")
    return passed, content

def llm_evaluate_with_retrieval(question, actual_answer, retrieved_context, expected_criteria):
    """
    Evaluates the agent's answer using an LLM judge, considering the retrieved context.
    Returns (bool, str): (passed, reason)
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
        
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    You are an impartial judge evaluating the performance of a movie question-answering bot.
    
    Question: {question}
    
    Context Retrieved by System (Movies found in database):
    {retrieved_context}
    
    Actual Answer: {actual_answer}
    
    Evaluation Criteria: 
    {expected_criteria}
    
    Based on the Context provided, is the Actual Answer reasonable and correct?
    Does the Actual Answer meet the evaluation criteria? 
    Respones with specific "YES" or "NO" on the first line, followed by a brief explanation.
    """
    
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    content = response.choices[0].message.content.strip()
    passed = content.upper().startswith("YES")
    return passed, content
