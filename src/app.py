import streamlit as st
import os
import json

from rag.pdf_processing import process_pdf
from rag.vector_store import add_documents_to_subject
from rag.chat import answer_question


st.set_page_config(
    page_title="Study Buddy",
)


if "subjects" not in st.session_state:

    if os.path.exists("subjects.json"):

        with open("subjects.json", "r") as file:
            st.session_state.subjects = json.load(file)

    else:

        st.session_state.subjects = ["Choose Subject"]


if "current_subject" not in st.session_state:
    st.session_state.current_subject = "Choose Subject"


if "messages" not in st.session_state:
    st.session_state.messages = []


st.sidebar.title("Study Buddy")

st.sidebar.write("Subjects")


selected_subject = st.sidebar.selectbox(
    "Select Subject",
    st.session_state.subjects
)


# Change subject
if selected_subject != st.session_state.current_subject:

    st.session_state.current_subject = selected_subject

    # Clear old chat
    st.session_state.messages = []

    st.rerun()


st.sidebar.write("Create New Subject")

new_subject = st.sidebar.text_input(
    "Enter subject name"
)


if st.sidebar.button("Add Subject"):

    if new_subject.strip() == "":

        st.sidebar.warning(
            "Please enter a subject name."
        )

    elif new_subject in st.session_state.subjects:

        st.sidebar.warning(
            "Subject already exists."
        )

    else:

        st.session_state.subjects.append(
            new_subject
        )

        with open(
            "subjects.json",
            "w"
        ) as file:

            json.dump(
                st.session_state.subjects,
                file
            )

        st.session_state.current_subject = (
            new_subject
        )

        st.session_state.messages = []

        st.rerun()



st.title(" Study Buddy")


if selected_subject == "Choose Subject":

    st.info(
        "Please select a subject from the sidebar."
    )

    st.stop()


st.header(
    "Subject: " + selected_subject
)


st.subheader("Choose Answer Type")


answer_type = st.selectbox(
    "How should the LLM answer?",
    [
        "Simple and Understandable",
        "10 Marks Detailed Answer"
    ]
)


st.subheader("Upload Study Material")


# SINGLE PDF UPLOAD

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    st.write(
        "Selected PDF: " +
        uploaded_file.name
    )


    if st.button("Process PDF"):

        # Create folder if it doesn't exist
        os.makedirs(
            "data/uploads",
            exist_ok=True
        )


        # Save PDF

        file_path = os.path.join(
            "data/uploads",
            uploaded_file.name
        )


        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )


        try:

            with st.spinner(
                "Processing PDF..."
            ):

                # PDF → Text → Chunks

                documents = process_pdf(
                    file_path
                )


                # Store in ChromaDB

                chunks = add_documents_to_subject(

                    documents=documents,

                    subject=selected_subject,

                    file_name=uploaded_file.name,

                    file_bytes=uploaded_file.getvalue()
                )


            st.success(
                uploaded_file.name +
                " added to " +
                selected_subject
            )


            st.write(
                "Number of chunks created:",
                chunks
            )


        except Exception as e:

            st.error(
                "Something went wrong while "
                "processing the PDF."
            )

            st.write(e)



st.divider()


st.subheader(
    "Ask " + selected_subject
)


# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# Chat input
question = st.chat_input(
    "Ask your question..."
)


if question:

    # Show user question
    with st.chat_message("user"):

        st.write(question)


    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                response = answer_question(

                    question=question,

                    subject=selected_subject,

                    answer_type=answer_type
                )


                st.write(response)


            except Exception as e:

                response = (
                    "Something went wrong: "
                    + str(e)
                )

                st.error(response)


    # Save AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )