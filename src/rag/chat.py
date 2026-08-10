from langchain_ollama import OllamaLLM

from .vector_store import search_subject


def get_llm():

    llm = OllamaLLM(
        model="llama3.1:8b",
        temperature=0
    )

    return llm


def answer_question(
    question,
    subject,
    answer_type
):

    # Search only the selected subject
    results = search_subject(
        question=question,
        subject=subject,
        k=4
    )

    if not results:

        return (
            "No information is available "
            "in the " +
            subject +
            " study material."
        )

    relevant_documents = []

    for document, score in results:

        if score >= 0.35:

            relevant_documents.append(
                document
            )


    if not relevant_documents:

        return (
            "I couldn't find this information "
            "in the " +
            subject +
            " study material."
        )


    context = ""

    for document in relevant_documents:

        context += (
            document.page_content
            + "\n\n"
        )


    if answer_type == "Simple and Understandable":

        answer_instructions = """

Give a simple and easy-to-understand answer.

Use simple English.

Explain the concept clearly.

Keep the answer reasonably short.

Use a small example if it helps.

Do not make the answer unnecessarily long.

"""


    else:

        answer_instructions = """

Give a detailed answer suitable for a
10-mark university examination.

Structure the answer properly.

Use headings and bullet points where useful.

Try to include:

1. Introduction / Definition
2. Detailed explanation
3. Important points
4. Example
5. Advantages / applications if relevant
6. Conclusion

The answer should be detailed enough
for a student to use it for a 10-mark answer.

"""

    prompt = f"""

You are Study Buddy.

Current Subject:
{subject}


IMPORTANT RULES:

1. Answer ONLY using the study material
   provided below.

2. Do NOT use your own general knowledge.

3. Do NOT use information from another subject.

4. Do NOT make up information.

5. If the answer is not present in the
   study material, say:

"No information available in the
uploaded study material."


ANSWER STYLE:

{answer_instructions}


STUDY MATERIAL:
----------------------------

{context}

----------------------------


STUDENT QUESTION:

{question}


ANSWER:
"""

    llm = get_llm()

    response = llm.invoke(
        prompt
    )

    return response