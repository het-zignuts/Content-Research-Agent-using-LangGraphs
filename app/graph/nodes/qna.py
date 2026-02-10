from app.llms.groq import get_groq_llm

def qna_node(state):
    QnA_PROMPT="""
        SYSTEM INSTRUCTIONS:
            You are a QnA assistant. Answer the question by asked in the user query based on the retrieved context provided below. 
            Use only the information from the retrieved context to answer the question. If you don't know the answer, say "I don't know. Couldn't figure it out from the context. Need more..."

        IMPORTANT INSTRUCTION: 
            For the answer generated, mention the source of the information in the retrieved context as source/document name and page number in the format [source: document_name, page: page_number]. 
            If the information is from multiple sources, mention all the sources in the same format separated by comma. 
            For example, if the answer is based on information from two documents "Doc1.pdf" page 2 and "Doc2.pdf" page 5, then the source should be mentioned as [source: Doc1.pdf, page: 2], [source: Doc2.pdf, page: 5].
            
        ANSWER FORMAT:
            Your answer should be in the following format:
            Answer: <your answer here>
            Citations: [source: document_name, page: page_number], [source: document_name, page: page_number], ...
            
        CONTEXT:
        {context}

        USER QUERY:
        {query}
    """
    llm=get_groq_llm(temperature=0.0)
    context="\n".join(f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Page Content: {d.page_content}\n" for d in state["documents"])
    answer=llm.invoke(
        QnA_PROMPT.format(context=context, query=state["query"])
    )

    return {"answer": answer}
