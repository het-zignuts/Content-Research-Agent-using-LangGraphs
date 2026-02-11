from app.llms.groq import get_groq_llm

def insight_node(state):
    """
    Insight Node for the LangGraph. This node takes the retrieved documents from the vector database and generates insights and recommendations based on the user query. It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. 
    The node then defines an insight generation prompt that instructs the LLM to analyze the retrieved content in relation to the user query and generate relevant, concise, and clear insights and recommendations. The LLM is invoked with this prompt, and the generated insights are returned in the state for downstream processing or response generation.
    """
    INSIGHT_PROMPT="""
        SYSTEM INSTRUCTION:
            You are an insight generation assistant.
            Analyze the retrieved content provided as context below, based on the user query, and generate recommendations and insights from the document content.
            Your insights should be relevant, concise and clear. 
            
        ANSWER FORMAT:
            Your answer should be in the following format:
            Insights & Recommendations: (your insights and recommendations here)
           
        CONTEXT:
        {context}

        USER QUERY:
        {query}
    """
    llm=get_groq_llm(temperature=0.3) # initialize the llm with 0.3 temperature to allow for some creativity in insights and recommendations
    #preparing the context
    context="\n".join(f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Page Content: {d.page_content}\n" for d in state["documents"])
    
    #invoke the llm with the formatted prompt, passing in the constructed context and the user query to generate insights and recommendations based on the provided documents and the user's request
    insight=llm.invoke(
        INSIGHT_PROMPT.format(context=context, query=state["query"])
    )
    return {"answer": insight} # add the answer to the state