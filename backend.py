from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.types import interrupt, Command

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool       #decorator

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

import math
import requests
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7)

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math functions.
    Input should be a valid math expression
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha vantage with API key in the URL.
    """

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=6BHRCWASB3CL161U"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol

    NOTE: This is a mock implementation simulating HITL:
    -Before confirming the purchase, this tool will interrupt
    -No real brokerage API is called
    -It returns a confirmation payload only
    """

    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase order for {quantity} shares of {symbol} was declined by the user.",
            "symbol": symbol,
            "quantity": quantity
        }


tools = [search_tool, calculator, get_stock_price, purchase_stock]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call"""
    messages = state['messages']
    
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)


connection = sqlite3.connect(database = "chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(connection)      

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
#graph.add_edge('chat_node', END)         reduntant

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    threads = checkpoint.list(None)   #I need all threads' information, not any specific thread

    all_threads = set()
    for thread in threads:
        all_threads.add(thread.config["configurable"]["thread_id"])

    return list(all_threads)



# initial_state = {
#     'messages': [HumanMessage(content="What is OOPs?")]
# }

# thread_id = "2"
# config = {'configurable': {'thread_id': thread_id}}
# result = chatbot.invoke(initial_state, config = config)
# print(result['messages'][-1].content)


# while True:
#     user_message = input("Type here: ")

#     print('User: ', user_message)
#     if user_message.strip().lower() in ['exit', 'quit', 'bye']:
#         break

#     config = {'configurable': {'thread_id': thread_id}}
#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
#     print("AI:  ", response['messages'][-1].content)




#-----------------------------
# Simple example for testing 
#-----------------------------

# if __name__ == "__main__":
#     print("Agentic chatbot CLI")
#     print("Type exit to quit. \n")

#     thread_id = "demo_thread"

#     while True:
#         user_input = input("You: ")
#         if user_input.lower().strip() in ["exit", "quit"]:
#             print("Bye")
#             break

#         #Build initial state for this turn
#         state = {"messages": [HumanMessage(content = user_input)]}

#         result = chatbot.invoke(state, 
#                                 config = {"configurable": {"thread_id" : thread_id}}
#                                 )

#         #check for HITL interrupt
#         interrupts = result.get("__interrupt__", [])

#         if interrupts:
#             #interrupt payload is the string we passed to the interrupt(...)
#             prompt_to_human = interrupts[0].value
#             print(f"HITL: {prompt_to_human}")
#             decision = input("Your decision: ").strip().lower()

#             result = chatbot.invoke(
#                 Command(resume = decision),
#                 config = {"configurable": {"thread_id": thread_id}}
#             )

#         #get latest message from assistant
#         messages = result["messages"]
#         last_message = messages[-1]
#         print(f"Bot: {last_message.content}\n")