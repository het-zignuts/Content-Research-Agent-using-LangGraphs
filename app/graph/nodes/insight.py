from app.llms.groq import get_groq_llm
from app.schemas.schemas import BaseSchema

def insight_node(state):
    """
    Insight Node for the LangGraph. This node takes the retrieved documents from the vector database and generates insights and recommendations based on the user query. It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. 
    The node then defines an insight generation prompt that instructs the LLM to analyze the retrieved content in relation to the user query and generate relevant, concise, and clear insights and recommendations. The LLM is invoked with this prompt, and the generated insights are returned in the state for downstream processing or response generation.
    """
    INSIGHT_PROMPT="""
        SYSTEM INSTRUCTION:
            
        1. You are an insight generation assistant in a LangGraph based Content research agent system.
            
        2. Analyze the retrieved content provided as context below, based on the user query, and generate recommendations or insights or opinions (whatever asked) from the document content.
            
        3. Your insights should be relevant, concise and clear. 

        4. In case you are not able generate response, or the given context does not contain the information required for the insight generation, STRICTLY return 
        "Couldn't provide insights from the given context..."

        5. Provide a concise plain-text answer in the 'answer' field with format "Insights and Recommendations: ....", as shown in example below:
        e.g.:   Query: What do you think should be the next step?
                Response:
                {{
                    "answer": "Insights and Recommendations: According to me, you should first communicate the details with the client..."
                }}

        The answer should not contain any other text than the actual answer.
        
        4. Follow this EXACT format with NO additional text before or after.

        5. The genrated response should be such that it ALWAYS survives pydantic model validation.
           
        CONTEXT:
        {context}

        USER QUERY:
        {query}
    """
    llm=get_groq_llm(temperature=0.3).with_structured_output(BaseSchema)
    
    context = "\n".join(
        f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Content: {d.page_content}\n" 
        for d in state["documents"]
    )
    
    try:
        response=llm.invoke(INSIGHT_PROMPT.format(context=context, query=state["query"]))
        return {
            "answer": response.answer,
        }
    except Exception as e:
        print(f"Insight Error: {str(e)}")
        return {
            "answer": "I encountered an error processing the documents.",
        }