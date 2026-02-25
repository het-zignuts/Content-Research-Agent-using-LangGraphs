from app.llms.groq import get_groq_llm
from app.schemas.schemas import BaseSchema

def qna_node(state):
    """
    QnA Node for the LangGraph. This node takes the retrieved documents from the vector database and answers the user query based on the retrieved context. 
    It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. The node then defines a QnA prompt that instructs the LLM to answer the user query using only the information from the retrieved context, and to provide citations for each piece of information used in the answer. The LLM is invoked with this prompt, and the generated answer is returned in the state for downstream processing or response generation.
    """
    QnA_PROMPT = """
        SYSTEM INSTRUCTIONS:
        You are a precise QnA assistant.

        1. Use ONLY the context provided below as reference to answer the questions.
        
        2. If you cannot generate response, or if the context provided is not sufficient to answer the query, then STRICTLY say "I don't know. Couldn't figure it out from the context.".
           Do not hallucinate.

        3. Provide a concise plain-text answer in the 'answer' field with format "Answer: .... Citations: ...", as shown in example below:
        e.g.:   Query: What does the policy say?
                Response:
                {{
                    "answer": "Answer: The policy says that she can get the money. \\n Citations: [source: Doc1.pdf, page: 2], [source: Doc2.pdf, page: 5]"
                }}

        The answer and citations should not contain any other text than the actual answer in the Answer field and citations as per above format in Citations field.

        4. Keep answer to 3-5 sentences maximum.
        
        5. Follow this EXACT format with NO additional text before or after.

        6. The genrated response should be such that it ALWAYS survives pydantic model validation.

        CONTEXT:
        {context}

        USER QUERY: 
        {query}
    """
    llm=get_groq_llm(temperature=0.0).with_structured_output(BaseSchema)
    
    context = "\n".join(
        f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Content: {d.page_content}\n" 
        for d in state["documents"]
    )
    
    try:
        response=llm.invoke(QnA_PROMPT.format(context=context, query=state["query"]))
        return {
            "answer": response.answer,
        }
    except Exception as e:
        print(f"QnA Error: {str(e)}")
        return {
            "answer": "I encountered an error processing the documents.",
        }