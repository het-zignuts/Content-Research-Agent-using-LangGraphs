from app.llms.groq import get_groq_llm
from app.schemas.schemas import *

def compare_node(state):
    """
    Compare Node for the LangGraph. This node takes the grouped documents retrieved from the vector database and compares them based on the user query. It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. The node then defines a comparison prompt that instructs the LLM to compare and contrast the documents in a structured tabular format, ensuring that each point of comparison is clearly identified along with its source. The LLM is invoked with this prompt, and the generated comparison is returned as the answer in the state for downstream processing or response generation.
    """

    COMPARISION_PROMPT = """
        SYSTEM INSTRUCTION:
        You are a compare assistant in a langgraph based content research agent system.

        1. Compare and contrast all the documents given in the context below, based on the user query.

        2. You MUST respond ONLY with a valid JSON object. No markdown code fences, no preamble, no extra text outside the JSON.

        3. The JSON must strictly follow this format:
        {{
            "answer": "<comparison_table_and_conclusion>"
        }}

        4. The value of "answer" must be a single string containing:
            a) A markdown table with columns: | Point of Comparison | <Doc1 Name> | <Doc2 Name> | ...
            - Each cell must be concise.
            - Each cell must include citation like: value (source: doc_name, page: N)
            - No excessive spaces. Table must be clean and properly formatted.
            b) After the table, a short conclusion starting on a new line: "Conclusion: ..."
            
        5. IMPORTANT: Since the answer is inside a JSON string:
            - Use \\n for newlines inside the answer string.
            - Do NOT use actual newline characters inside the JSON string value.
            - Escape any special characters properly.

        6. Do NOT mix facts across documents.
        7. Do NOT hallucinate. STRICTLY adhere to the provided context only.
        8. If the query is out of context or unanswerable from provided docs, the answer field must be exactly:
        "Couldn't generate comparison, based on provided context..."

        EXAMPLE OF VALID RESPONSE (structure only, not real data):
        {{"answer": "| Point of Comparison | Phone A | Phone B |\\n|---|---|---|\\n| Display | 6.1 inch OLED (source: docA, page: 1) | 6.5 inch LCD (source: docB, page: 2) |\\n| Battery | 4000mAh (source: docA, page: 2) | 4500mAh (source: docB, page: 3) |\\n\\nConclusion: Phone B has a larger battery while Phone A offers a superior display technology."}}

        CONTEXT:
        {context}

        USER QUERY:
        {query}
    """

    llm=get_groq_llm(temperature=0.0).with_structured_output(ComparisonResponse)  # initialize the GROQ LLM with a temperature of 0.0 for deterministic output.

    docs=state["grouped_docs"] # get the grouped docs from the sate
    context="" # initialize an empty string to build the context for the LLM prompt
    for doc_id, contents in docs.items():
        doc_name=contents[0]["doc_name"] # retrieve documant name
        context+=f"Document Name: {doc_name}\n"
        context+=f"Context from {doc_name}:\n"
        for content in contents:
            context+=f"- {content['content']}\n Page Number: {content['page_number']}\n" # add each content piece from the document to the context, along with its page number for reference

    try:
        response=llm.invoke(COMPARISION_PROMPT.format(context=context, query=state["query"])) # Invoke the LLM with the formatted prompt, passing in the constructed context and the user query to generate the comparison output based on the provided documents and the user's request
        return {
            "answer": response.answer, # return the generated comparison as the answer in the state, which can be used by downstream nodes in the graph to provide a response to the user or for further processing.
        }
    except Exception as e:
        print(f"Comparison Error: {str(e)}")
        return {
            "answer": "I encountered an error processing the documents.",
        }