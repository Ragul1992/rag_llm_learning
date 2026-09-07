"""Phase 1: retrieve relevant chunks and generate a grounded answer."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "phase1_rag"


def get_vectorstore() -> Chroma:
    """Connect to the existing persistent Chroma collection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(question: str, k: int = 4) -> list:
    """Retrieve the top-k semantically similar chunks."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(question, k=k)


def format_context(docs: list) -> str:
    """Format retrieved chunks for the LLM prompt."""
    if not docs:
        return "No relevant context was found."

    sections = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        sections.append(
            f"[Source {i}: {source}{page_text}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(sections)


def ask(question: str) -> str:
    """Run the basic 2-step RAG pipeline."""
    docs = retrieve(question, k=4)
    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a document question-answering assistant.
Answer ONLY from the provided context.
If the context does not contain the answer, say that the answer was not found in the documents.
Do not invent facts.
Mention the relevant source names in your answer.

Context:
{context}""",
            ),
            ("human", "Question: {question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    print("\n=== RETRIEVED CHUNKS ===")
    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i} | {doc.metadata}")
        print(doc.page_content[:500].replace("\n", " "))

    print("\n=== ANSWER ===")
    print(response.content)

    return response.content


if __name__ == "__main__":
    question = input("\nAsk a question: ").strip()
    if not question:
        raise SystemExit("Question cannot be empty.")

    if not CHROMA_DIR.exists():
        raise SystemExit("Vector DB not found. Run: python ingest.py")

    ask(question)
