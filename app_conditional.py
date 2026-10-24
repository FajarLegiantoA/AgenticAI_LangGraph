from typing import Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph

# Use the model client that was successfully configured
# in our first application.
from app import llm


# 1. Inisialisasi Model Embedding Qwen3 via Hugging Face
print("Loading Qwen3 Embedding Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cpu"},  # Ubah ke 'cuda' jika menggunakan GPU Nvidia
    encode_kwargs={"normalize_embeddings": True},
)

# 2. Definisikan Prototipe Pertanyaan untuk Setiap Rute
routing_samples = [
    # Samples for Coding / SQL
    Document(page_content="Write a SQL query with SELECT, JOIN, or GROUP BY", metadata={"route": "coding"}),
    Document(page_content="How do I filter and aggregate database tables in Postgres or MySQL?", metadata={"route": "coding"}),
    Document(page_content="Debug code, fix syntax errors, and resolve exceptions", metadata={"route": "coding"}),
    Document(page_content="Write a Python or JavaScript function, algorithm, or script", metadata={"route": "coding"}),
    Document(page_content="How do I connect to a relational database using an ORM?", metadata={"route": "coding"}),

    # Samples for AI / ML
    Document(page_content="Explain deep learning architecture, transformers, or neural networks", metadata={"route": "ml"}),
    Document(page_content="How does Retrieval-Augmented Generation (RAG) and vector embedding work?", metadata={"route": "ml"}),
    Document(page_content="Fine-tune Large Language Models (LLM) and train machine learning models", metadata={"route": "ml"}),
    Document(page_content="Backpropagation, loss functions, and gradient descent optimization", metadata={"route": "ml"}),
    Document(page_content="Difference between supervised learning, unsupervised learning, and reinforcement learning", metadata={"route": "ml"}),

    # Samples for General Tech / Explanations
    Document(page_content="Explain how the internet, DNS, and IP addresses work to a beginner", metadata={"route": "general"}),
    Document(page_content="What is cloud computing, virtualization, and how does the web function?", metadata={"route": "general"}),
    Document(page_content="History of computer hardware, CPUs, and operating systems", metadata={"route": "general"}),
    Document(page_content="General tips for productivity and basic computer usage", metadata={"route": "general"}),
]

# 3. Buat Vector Store Chroma di dalam Memori
vector_store = Chroma.from_documents(
    documents=routing_samples,
    embedding=embeddings,
    collection_name="intent_router",
)


# 1. Define graph state
class GraphState(TypedDict):
    question: str
    route: str
    answer: str


# 2. Node to classify the question category
def classify_question(state: GraphState) -> dict:
    query = state["question"]

    # Cari 1 dokumen prototipe yang paling mirip secara makna/semantik
    results = vector_store.similarity_search(query, k=1)

    if results:
        route = results[0].metadata.get("route", "general")
        matched_sample = results[0].page_content
    else:
        route = "general"
        matched_sample = "None"

    print(f"\n[Semantic Router] Question : '{query}'")
    print(f"[Semantic Router] Matched  : '{matched_sample}'")
    print(f"[Semantic Router] Target   : {route}\n")

    return {"route": route}


# 3. Node specifically for coding and SQL questions
def coding_node(state: GraphState) -> dict:
    response = llm.invoke(
        [
            (
                "system",
                "You are a senior software engineer and database engineer. "
                "Answer coding and SQL queries with efficient code/SQL examples "
                "and concise explanations in English.",
            ),
            ("human", state["question"]),
        ]
    )

    return {"answer": str(response.content)}


# 4. Node specifically for AI / Machine Learning questions
def ml_node(state: GraphState) -> dict:
    response = llm.invoke(
        [
            (
                "system",
                "You are an AI/ML research scientist and engineer. "
                "Explain Machine Learning, Neural Networks, LLMs, and AI concepts "
                "clearly with practical insights or code snippets in English.",
            ),
            ("human", state["question"]),
        ]
    )

    return {"answer": str(response.content)}


# 4. Node specifically for general questions
def general_node(state: GraphState) -> dict:
    response = llm.invoke(
        [
            (
                "system",
                "You are an AI assistant who explains general technology "
                "concepts in simple, clear, and beginner-friendly English.",
            ),
            ("human", state["question"]),
        ]
    )

    return {"answer": str(response.content)}


# 5. Function to select the execution route
def choose_route(
    state: GraphState,
) -> Literal["coding", "ml", "general"]:
    return state["route"]


# 6. Assemble the graph
builder = StateGraph(GraphState)

builder.add_node("classify", classify_question)
builder.add_node("coding_node", coding_node)
builder.add_node("ml_node", ml_node)
builder.add_node("general_node", general_node)

builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    choose_route,
    {
        "coding": "coding_node",
        "ml": "ml_node",
        "general": "general_node",
    },
)

builder.add_edge("coding_node", END)
builder.add_edge("ml_node", END)
builder.add_edge("general_node", END)

graph = builder.compile()


# 7. Run the terminal application
def main():
    print("=== LangGraph Conditional Routing ===")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = graph.invoke(
                {
                    "question": question,
                    "route": "",
                    "answer": "",
                }
            )

            print(f"\nRoute : {result['route']}")
            print(f"AI    : {result['answer']}\n")

        except Exception as exc:
            print(f"\nAn error occurred: {exc}\n")

print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    main()