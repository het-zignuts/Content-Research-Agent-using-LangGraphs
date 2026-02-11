from app.llms.groq import get_groq_llm

def extract_node(state):
    EXTRACTION_PROMPT="""
        SYSTEM INSTRUCTION:
            You are an info-extraction and report generation assistant.
            Extract the key information from the retrieved content provided as context below, based on the user query.
            Then, generate a research report in markdown format based on the extracted information.

            Return only the .md format report as a string, for example:
            report=f"# Research Report .... "

            No other text format should be returned other than the markdown report.

        CONTEXT:
        {context}

        USER QUERY:
        {query}
    """

    llm=get_groq_llm(temperature=0.0)
    context="\n".join(f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Page Content: {d.page_content}\n" for d in state["documents"])
    report=llm.invoke(
        EXTRACTION_PROMPT.format(context=context, query=state["query"])
    )
    return {"report_md": report.content}