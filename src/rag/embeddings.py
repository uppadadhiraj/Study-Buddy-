from langchain_ollama import OllamaEmbeddings


def get_embedding_model():
    """
    Returns the Ollama embedding model.
    """

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    return embeddings