
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

# Use the model client that you have successfully tested.
from app import llm


# 1. Create a multiplication tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers. Use this to calculate a times b."""
    result = a * b
    print(f"[Tool] multiply({a}, {b}) = {result}")
    return result


# 2. Create an addition tool
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    result = a + b
    print(f"[Tool] add({a}, {b}) = {result}")
    return result


# 3. Register all tools the model is allowed to use
tools = [multiply, add]

# Bind the list of tools to the model.
# The model must support tool/function calling.
llm_with_tools = llm.bind_tools(tools)


# 4. Define the node to call the model
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])

    # MessagesState will append the new response message
    # to the existing messages.
    return {"messages": [response]}


# 5. Build the graph
builder = StateGraph(MessagesState)

builder.add_node("llm", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "llm")

# Jika model meminta tool, jalankan ToolNode.
# Jika tidak, akhiri graph.
builder.add_conditional_edges("llm", tools_condition)

# After the tool finishes, the model receives the output
# and can generate the final answer.
builder.add_edge("tools", "llm")

graph = builder.compile()


# 6. Run the application via the terminal
def main():
    print("=== LangGraph Tool Calling ===")
    print("Available tools: multiply, add")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = graph.invoke({
                "messages": [HumanMessage(content=question)]
            })

            final_message = result["messages"][-1]
            print(f"\nAI: {final_message.content}\n")

        except Exception as exc:
            print(f"\nAn error occurred: {exc}\n")


if __name__ == "__main__":
    main()
    