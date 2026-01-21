from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import os

def llm_evaluate(question, actual_answer, expected_criteria):
    """
    Evaluates the agent's answer using an LLM judge.
    Returns (bool, str): (passed, reason)
    """
    
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY not found")
        
    llm = ChatOllama(model="gemma3:12b", temperature=0)
    
    prompt = f"""
    You are an impartial judge evaluating the performance of a movie question-answering bot.
    
    Question: {question}
    
    Actual Answer: {actual_answer}
    
    Evaluation Criteria: 
    {expected_criteria}
    
    Does the Actual Answer meet the evaluation criteria? 
    Respones with specific "YES" or "NO" on the first line, followed by a brief explanation.
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    content = response.content.strip()
    passed = content.upper().startswith("YES")
    return passed, content

def llm_evaluate_with_retrieval(question, actual_answer, retrieved_context, expected_criteria):
    """
    Evaluates the agent's answer using an LLM judge, considering the retrieved context.
    Returns (bool, str): (passed, reason)
    """
    
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY not found")
        
    llm = ChatOllama(model="gemma3:12b", temperature=0)
    
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
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    content = response.content.strip()
    passed = content.upper().startswith("YES")
    return passed, content
