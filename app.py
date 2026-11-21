import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from backend import chatbot, get_all_threads


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat() -> None:
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def get_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace",
    }


def load_conversation(thread_id: str) -> list:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_pending_interrupt(thread_id: str) -> str | None:
    """Return the interrupt prompt string if the graph is currently paused on an interrupt."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    for task in getattr(state, "tasks", ()):
        interrupts = getattr(task, "interrupts", ())
        if interrupts:
            return interrupts[0].value
    return None


def messages_to_ui_history(messages: list) -> list[dict]:
    temp_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "assistant"
            if message.tool_calls:
                calls_str = ", ".join(
                    f"`{tc['name']}` with args `{tc['args']}`" for tc in message.tool_calls
                )
                content = f"🛠️ **Calling tool:** {calls_str}"
                if message.content:
                    content = f"{message.content}\n\n{content}"
            else:
                content = message.content
        else:
            continue

        if content:
            temp_messages.append({"role": role, "content": content})
    return temp_messages


def run_and_stream(graph_input, config: dict) -> None:
    """Stream model/tool output and append final assistant text if a turn completes."""
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _metadata in chatbot.stream(
                graph_input,
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    # Gemini content can occasionally be a list of blocks; normalize to str
                    if isinstance(message_chunk.content, list):
                        text_parts = [
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in message_chunk.content
                        ]
                        yield "".join(text_parts)
                    else:
                        yield str(message_chunk.content)

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    # Only save a final assistant message if execution didn't pause on a new interrupt
    if get_pending_interrupt(config["configurable"]["thread_id"]) is None and ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )


st.title("Agentic chatbot with langgraph")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

add_thread(st.session_state["thread_id"])

# ===================== Sidebar =====================
st.sidebar.title("My conversations")

if st.sidebar.button("New chat"):
    reset_chat()
    st.rerun()

for tid in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(tid), key=f"thread_{tid}"):
        st.session_state["thread_id"] = tid
        messages = load_conversation(tid)
        st.session_state["message_history"] = messages_to_ui_history(messages)
        st.rerun()

# ===================== Chat History =====================
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

current_thread_id = st.session_state["thread_id"]
config = get_config(current_thread_id)
pending_interrupt = get_pending_interrupt(current_thread_id)

# ===================== HITL Approval UI =====================
if pending_interrupt is not None:
    with st.chat_message("assistant"):
        st.warning(f"⏸️ **Approval Required:** {pending_interrupt}")
        col1, col2 = st.columns(2)
        approve_clicked = col1.button("✅ Approve (yes)", key=f"approve_{current_thread_id}")
        reject_clicked = col2.button("❌ Reject (no)", key=f"reject_{current_thread_id}")

    if approve_clicked or reject_clicked:
        decision = "yes" if approve_clicked else "no"
        run_and_stream(Command(resume=decision), config=config)
        st.rerun()

# ===================== Chat Input =====================
user_input = st.chat_input(
    "Approve or reject the pending action above first..."
    if pending_interrupt is not None
    else "Type here"
)

if user_input and pending_interrupt is None:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    run_and_stream({"messages": [HumanMessage(content=user_input)]}, config=config)

    # If a new interrupt was triggered during this turn, rerun so the approval buttons render immediately
    if get_pending_interrupt(current_thread_id) is not None:
        st.rerun()