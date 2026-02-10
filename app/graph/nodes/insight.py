from app.llms.groq import get_groq_llm

def insight_node(state):
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
    llm=get_groq_llm(temperature=0.3)
    context="\n".join(f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Page Content: {d.page_content}\n" for d in state["documents"])
    insight=llm.invoke(
        INSIGHT_PROMPT.format(context=context, query=state["query"])
    )
    return {"answer": insight}