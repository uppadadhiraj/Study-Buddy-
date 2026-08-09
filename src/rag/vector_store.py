import hashlib

from langchain_chroma import Chroma
from .embeddings import get_embedding_model


CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "study_buddy"


def get_vector_store():
    """
    Creates/loads the persistent ChromaDB vector store.
    """

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vector_store


def add_documents_to_subject(
    documents,
    subject,
    file_name,
    file_bytes
):
    """
    Stores PDF chunks in ChromaDB.

    Every chunk receives metadata containing the subject.
    """

    vector_store = get_vector_store()

    # Create a unique ID for this PDF
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    document_ids = []

    for index, document in enumerate(documents):

        # VERY IMPORTANT:
        # Store the subject as metadata.
        document.metadata["subject"] = subject
        document.metadata["file_name"] = file_name
        document.metadata["file_hash"] = file_hash

        # Unique ID for every chunk
        chunk_id = f"{subject}_{file_hash}_{index}"

        document_ids.append(chunk_id)

    vector_store.add_documents(
        documents=documents,
        ids=document_ids
    )

    return len(documents)


def search_subject(
    question,
    subject,
    k=4
):
    """
    Searches ONLY inside the selected subject.
    """

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_relevance_scores(
        query=question,
        k=k,
        filter={
            "subject": subject
        }
    )

    return results