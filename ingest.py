"""Phase 1: ingest PDF documents into a local Chroma vector database."""

from pathlib import Path
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "phase1_rag"


def load_pdfs() -> list:
    """Load every PDF in data/documents."""
    documents = []
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {DOCUMENTS_DIR}. Add at least one PDF and run again."
        )

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        documents.extend(docs)

    return documents


def split_documents(documents: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def create_vectorstore(chunks: list) -> Chroma:
    """Create a persistent Chroma collection and add chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # For Phase 1 we rebuild the local DB each time to avoid duplicate documents.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vectorstore.add_documents(chunks)
    return vectorstore


def main() -> None:
    print("\n=== PHASE 1: DOCUMENT INGESTION ===\n")

    documents = load_pdfs()
    print(f"Loaded pages: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    create_vectorstore(chunks)
    print(f"Vector database ready: {CHROMA_DIR}")

    print("\nSample chunk:")
    print("-" * 80)
    print(chunks[0].page_content[:800])
    print("-" * 80)
    print(chunks[0].metadata)


if __name__ == "__main__":
    main()
