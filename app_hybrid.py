
import re

from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# Gunakan model client yang sudah berhasil kamu uji.
from app import llm

from profile_memory import (
    init_profile_db,
    get_profile,
    update_profile_name,
)

load_dotenv()


# =========================================================
# 1. SEMANTIC ROUTING: QWEN3 + CHROMA
# =========================================================

print("Loading Qwen3 Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


routing_samples = [
    # Coding / SQL
    Document(
        page_content="Write SQL queries using SELECT, JOIN, GROUP BY, "
                     "ORDER BY, and database aggregation.",
        metadata={"route": "coding"},
    ),
    Document(
        page_content="Find the five highest paid employees using SQL "
                     "queries in PostgreSQL or MySQL.",
        metadata={"route": "coding"},
    ),
    Document(
        page_content="How do I filter and aggregate database tables "
                     "in Postgres or MySQL?",
        metadata={"route": "coding"},
    ),
    Document(
        page_content="Debug Python code, fix syntax errors, and "
                     "write functions or algorithms.",
        metadata={"route": "coding"},
    ),
    Document(
        page_content="Write JavaScript code, scripts, and software "
                     "applications.",
        metadata={"route": "coding"},
    ),
    Document(
        page_content="How do I connect to a relational database using "
                     "an ORM?",
        metadata={"route": "coding"}
    ),

    # AI / ML
    Document(
        page_content="Explain machine learning, neural networks, "
                     "deep learning, and model training.",
        metadata={"route": "ml"},
    ),
    Document(
        page_content="Explain transformer architecture, attention, "
                     "LLMs, and tokenization.",
        metadata={"route": "ml"},
    ),
    Document(
        page_content="Explain RAG, vector embeddings, vector databases, "
                     "and semantic search.",
        metadata={"route": "ml"},
    ),
    Document(
        page_content="How does Retrieval-Augmented Generation (RAG) and "
        "vector embedding work?",
        metadata={"route": "ml"}
    ),
    Document(
        page_content="Explain gradient descent, loss functions, "
                     "backpropagation, and fine-tuning.",
        metadata={"route": "ml"},
    ),
    Document(
        page_content="Difference between supervised learning, unsupervised "
                     "learning, and reinforcement learning",
        metadata={"route": "ml"}
    ),

    # General technology
    Document(
        page_content="Explain how the internet, DNS, IP addresses, "
                     "and computer networks work for beginners.",
        metadata={"route": "general"},
    ),
    Document(
        page_content="Explain cloud computing, operating systems, "
                     "virtualization, and computer hardware.",
        metadata={"route": "general"},
    ),
    Document(
        page_content="Explain general technology concepts in simple "
                     "language for non-technical users.",
        metadata={"route": "general"},
    ),
    Document(
        page_content="Explain how websites, web browsers, and HTTP "
                     "work in everyday life.",
        metadata={"route": "general"},
    ),
    Document(
        page_content="History of computer hardware, CPUs, and operating "
                     "systems",
        metadata={"route": "general"}
    )
]

print("Building routing vector store...")

vector_store = Chroma.from_documents(
    documents=routing_samples,
    embedding=embeddings,
    collection_name="hybrid_intent_router",
)


def semantic_route(question: str) -> tuple[str, float]:
    """Pilih route berdasarkan contoh dengan jarak embedding terdekat."""

    matches = vector_store.similarity_search_with_score(
        question,
        k=3,
    )

    if not matches:
        return "general", float("inf")

    # Pada Chroma, score dari similarity_search_with_score
    # umumnya merupakan jarak: semakin kecil, semakin dekat.
    best_document, best_distance = matches[0]

    route = best_document.metadata.get("route", "general")

    print(f"\n[Router] Best route : {route}")
    print(f"[Router] Distance  : {best_distance:.4f}")

    print("[Router] Top matches:")
    for document, distance in matches:
        print(
            f"  - {document.metadata.get('route')} "
            f"| distance={distance:.4f} "
            f"| {document.page_content[:75]}"
        )

    return route, float(best_distance)


# =========================================================
# 2. TOOLS
# =========================================================

# 1. Create a multiplication tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers. Use this to calculate a times b."""
    result = a * b
    print(f"\n[Tool] multiply({a}, {b}) = {result}")
    return result


# 2. Create an addition tool
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    result = a + b
    print(f"\n[Tool] add({a}, {b}) = {result}")
    return result

# 3. Register all tools the model is allowed to use
tools = [multiply, add]

# Bind the list of tools to the model.
# The model must support tool/function calling.
llm_with_tools = llm.bind_tools(tools)


# =========================================================
# 3. GRAPH STATE
# =========================================================

class AgentState(MessagesState):
    route: str
    route_distance: float


# =========================================================
# 4. SEMANTIC ROUTER NODE
# =========================================================

def router_node(state: AgentState) -> dict:
    question = next(
        message.content
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )

    route, distance = semantic_route(str(question))

    return {
        "route": route,
        "route_distance": distance,
    }


# =========================================================
# 5. LLM NODE
# =========================================================

SYSTEM_PROMPTS = {
    "coding": (
        "You are a senior software engineer and database engineer. "
        "Answer coding and SQL questions accurately. "
        "Use tools when they are genuinely helpful. "
        "Respond in the language used by the user."
    ),
    "ml": (
        "You are an AI and machine learning engineer and research scientist. "
        "Explain embeddings, machine learning, neural networks, "
        "and LLM concepts accurately. "
        "Use tools when they are genuinely helpful. "
        "Respond in the language used by the user."
    ),
    "general": (
        "You are a helpful technology assistant. "
        "Explain concepts clearly and adjust explanations "
        "to the user's level. Use tools when genuinely helpful. "
        "Respond in the language used by the user."
    ),
}


def call_model(state: AgentState) -> dict:
    route = state.get("route", "general")
    system_prompt = SYSTEM_PROMPTS.get(
        route,
        SYSTEM_PROMPTS["general"],
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
    )

    return {"messages": [response]}


# =========================================================
# 6. BUILD LANGGRAPH
# =========================================================

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("llm", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "router")
builder.add_edge("router", "llm")

# Model memilih: panggil tool atau selesai.
builder.add_conditional_edges("llm", tools_condition)

# Setelah tool selesai, kembali ke model.
builder.add_edge("tools", "llm")


# =========================================================
# 7. INTERACTIVE CLI
# =========================================================

def main():
    print("\n=== LangGraph Hybrid Agent ===")
    print("Semantic routing + Tool Calling")
    print("Routes: coding, ml, general")
    print("Tools: add, multiply")
    print("Ketik 'exit' untuk keluar.\n")

    # Inisialisasi database long-term memory.
    init_profile_db()

    # Fixed user ID for local experimentation.
    # Distinct from thread_id, which uniquely identifies a conversation session.
    user_id = "local-user"

    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        
        while True:
            question = input("You: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not question:
                continue

            try:
                # 1. Fetch the user's long-term memory profile.
                profile = get_profile(user_id)

                # 2. Detect simple name introductions.
                # Examples: "My name is Fajar" or "Nama saya Fajar".
                match = re.search(
                    r"^\s*(?:my name is|nama saya|nama aku)\s+"
                    r"([A-Za-z][A-Za-z '\-]{0,79})\s*[.!]?\s*$",
                    question,
                    flags=re.IGNORECASE,
                )

                if match:
                    name = match.group(1).strip().rstrip(".!").strip()

                    if name:
                        update_profile_name(user_id, name)
                        profile = get_profile(user_id)
                        print(
                            f"[Profile Memory] Name '{name}' "
                            "successfully saved."
                        )

                # 3. Inject profile context into the model prompt.
                # This context is provided only to the model for this request;
                # it does not alter the user message saved in the graph state.
                if profile and profile.get("name"):
                    memory_context = (
                        "Long-term user profile:\n"
                        f"- User's name: {profile['name']}\n"
                        "Use this profile when relevant to the user's "
                        "question. Do not claim that the user said this "
                        "in the current conversation."
                    )
                    message = HumanMessage(
                        content=(
                            f"{memory_context}\n\n"
                            f"User's current message:\n{question}"
                        )
                    )
                    print(
                        f"[Profile Memory] Saved name: "
                        f"{profile['name']}"
                    )
                else:
                    message = HumanMessage(content=question)

                # 4. Set execution thread configuration.
                # Changing thread_id creates an isolated state checkpoint.
                config = {
                    "configurable": {
                        "thread_id": "learning-session-3"
                    }
                }

                result = graph.invoke(
                    {
                        "messages": [message]
                    },
                    config=config,
                )

                final_message = result["messages"][-1]

                print(f"\nRoute: {result['route']}")
                print(f"AI: {final_message.content}\n")

                # 5. Debug metadata IFM
                for i, message in enumerate(result["messages"]):
                    if isinstance(message, AIMessage):
                        print(f"[Debug] Assistant message #{i}")
                        print(
                            "[Debug] additional_kwargs keys:",
                            list(message.additional_kwargs.keys()),
                        )
                        print(
                            "[Debug] reasoning_content present:",
                            "reasoning_content" in message.additional_kwargs,
                        )
                        print(
                            "[Debug] response_metadata keys:",
                            list(message.response_metadata.keys()),
                        )

            except Exception as exc:
                print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
