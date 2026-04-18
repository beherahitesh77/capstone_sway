# ============================================================
# capstone_streamlit.py — HR Policy Assistant
# Run: streamlit run capstone_streamlit.py
# ============================================================

import streamlit as st
import uuid
import os
import calendar
from datetime import date, timedelta
from dotenv import load_dotenv
from typing import TypedDict, List

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import numpy as np

class NumpyCollection:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
    def add(self, documents, embeddings, ids, metadatas):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)
        self.np_embeddings = np.array(self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.np_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.np_embeddings = self.np_embeddings / norms
        
    def query(self, query_embeddings, n_results=3):
        q_emb = np.array(query_embeddings, dtype=np.float32)
        q_norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1e-10
        q_emb = q_emb / q_norms
        similarities = np.dot(q_emb, self.np_embeddings.T)
        out_docs = []
        out_meta = []
        for i in range(len(similarities)):
            sim = similarities[i]
            top_indices = np.argsort(sim)[::-1][:n_results]
            out_docs.append([self.documents[idx] for idx in top_indices])
            out_meta.append([self.metadatas[idx] for idx in top_indices])
        return {"documents": out_docs, "metadatas": out_meta}
    
    def count(self):
        return len(self.documents)

load_dotenv()

st.set_page_config(
    page_title="HR Policy Assistant", page_icon="🏢", layout="centered"
)
st.title("🏢 HR Policy Assistant")
st.caption("AI-powered HR research 24/7 for employees.")

# ============================================================
# KNOWLEDGE BASE
# ============================================================
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Paid Time Off (PTO)",
        "text": "All full-time employees accrue 15 days of Paid Time Off (PTO) annually. PTO accrues per pay period. Employees may carry over up to 5 days of unused PTO to the next calendar year. Any additional unused PTO will be forfeited. PTO must be requested at least two weeks in advance for planned vacations."
    },
    {
        "id": "doc_002",
        "topic": "Sick Leave",
        "text": "Employees receive 10 days of paid sick leave per calendar year. Sick leave does not carry over. If an employee is out sick for more than 3 consecutive days, a doctor's note may be required before returning to work."
    },
    {
        "id": "doc_003",
        "topic": "Payroll Details",
        "text": "Employees are paid on a semi-monthly basis. Paydays occur on the 15th and the last day of each month. If a payday falls on a weekend or company holiday, direct deposits will be processed on the preceding business day. Overtime for non-exempt employees must be pre-approved by a manager."
    },
    {
        "id": "doc_004",
        "topic": "Remote Work Policy",
        "text": "The company operates on a hybrid model. Employees are eligible to work remotely for up to 2 days per week with their manager's approval. Employees must ensure their remote workspace has a stable internet connection and is free from distractions during core working hours."
    },
    {
        "id": "doc_005",
        "topic": "Core Working Hours",
        "text": "Our core working hours are 10:00 AM to 3:00 PM in the employee's local time zone. During this window, employees are expected to be online, responsive, and available for meetings. Outside of these hours, employees have flexibility to structure their 40-hour work week."
    },
    {
        "id": "doc_006",
        "topic": "Performance Reviews",
        "text": "Annual performance reviews occur every December. Merit increases, if awarded, take effect in the first pay period of February. There is also a formal mid-year check-in during June to discuss goals and progression."
    },
    {
        "id": "doc_007",
        "topic": "Bereavement Leave",
        "text": "Employees are entitled to up to 3 days of paid bereavement leave in the event of the death of an immediate family member (spouse, child, parent, sibling, grandparent). Additional unpaid leave may be granted at the manager's discretion."
    },
    {
        "id": "doc_008",
        "topic": "Parental Leave",
        "text": "The company offers 12 weeks of fully paid parental leave for new parents following the birth, adoption, or foster placement of a child. This leave can be taken consecutively or continuously within the first year of the child's arrival."
    },
    {
        "id": "doc_009",
        "topic": "Code of Conduct",
        "text": "The company maintains a zero-tolerance policy against harassment, discrimination, and retaliation. We are committed to a safe, inclusive workplace. Any violations should be reported immediately to HR or anonymously via the ethics hotline."
    },
    {
        "id": "doc_010",
        "topic": "Company Holidays",
        "text": "The company observes 10 paid holidays each year: New Year's Day, Martin Luther King Jr. Day, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving Day, the day after Thanksgiving, Christmas Eve, and Christmas Day."
    }
]

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


# ============================================================
# LOAD AGENT (cached — only runs once per session)
# ============================================================
@st.cache_resource
def load_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    collection = NumpyCollection()

    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    # ── Node definitions ───────────────────────────────────
    class CapstoneState(TypedDict):
        question: str
        messages: List[dict]
        route: str
        retrieved: str
        sources: List[str]
        tool_result: str
        answer: str
        faithfulness: float
        eval_retries: int

    def memory_node(state):
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    def router_node(state):
        question = state["question"]
        messages = state.get("messages", [])
        recent = (
            "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1])
            or "none"
        )
        prompt = f"""You are a router for an HR Policy Assistant used by employees.
Available routes:
- retrieve: search the company handbook (leave, payroll, code of conduct, benefits, remote work)
- memory_only: answer from conversation history (e.g. 'what did you just say?')
- tool: use the dates tool for questions about today's date, paydays, or company holidays

Recent conversation: {recent}
Current question: {question}
Reply with ONLY one word: retrieve / memory_only / tool"""
        decision = llm.invoke(prompt).content.strip().lower()
        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        else:
            decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        today = date.today()
        lines = [f"Today's date: {today.strftime('%B %d, %Y')} ({today.isoformat()})"]
        lines.append("\nUpcoming standard dates from today:")
        
        payday1 = today.replace(day=15)
        if today.day > 15:
            if today.month == 12:
                payday1 = today.replace(year=today.year+1, month=1, day=15)
            else:
                payday1 = today.replace(month=today.month+1, day=15)
                
        _, last_day = calendar.monthrange(today.year, today.month)
        payday2 = today.replace(day=last_day)
        if today.day == last_day:
            _, next_last_day = calendar.monthrange(today.year + (today.month // 12), (today.month % 12) + 1)
            payday2 = payday2.replace(day=next_last_day, month=(today.month % 12) + 1, year=today.year + (today.month // 12))
            
        days_left1 = (payday1 - today).days
        days_left2 = (payday2 - today).days

        lines.append(f"  • Upcoming mid-month payday: {payday1.strftime('%B %d, %Y')} ({days_left1} days away)")
        lines.append(f"  • Upcoming end-of-month payday: {payday2.strftime('%B %d, %Y')} ({days_left2} days away)")
        
        return {"tool_result": "\n".join(lines)}

    def answer_node(state):
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        context_parts = []
        if retrieved:
            context_parts.append(f"COMPANY HANDBOOK:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TODAY'S DATE & PAYDAYS:\n{tool_result}")
        context = "\n\n".join(context_parts)
        if context:
            system_content = f"""You are an HR Policy Assistant helping company employees.
Answer using ONLY the information provided in the context below.
If the answer is not in the context, say: I don't have that information in my knowledge base.
Do NOT add information from your training data.

{context}"""
        else:
            system_content = "You are an HR Policy Assistant. Answer based on the conversation history."
        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: Answer using ONLY information explicitly stated in the context above."
        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(
                HumanMessage(content=msg["content"])
                if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
        lc_msgs.append(HumanMessage(content=question))
        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    def eval_node(state):
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)
        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
Context: {context}
Answer: {answer[:300]}"""
        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state):
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    def route_decision(state):
        route = state.get("route", "retrieve")
        if route == "tool":
            return "tool"
        if route == "memory_only":
            return "skip"
        return "retrieve"

    def eval_decision(state):
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    # ── Assemble graph ─────────────────────────────────────
    g = StateGraph(CapstoneState)
    g.add_node("memory", memory_node)
    g.add_node("router", router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("skip", skip_retrieval_node)
    g.add_node("tool", tool_node)
    g.add_node("answer", answer_node)
    g.add_node("eval", eval_node)
    g.add_node("save", save_node)
    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    g.add_edge("retrieve", "answer")
    g.add_edge("skip", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    g.add_edge("save", END)

    agent_app = g.compile(checkpointer=MemorySaver())
    return agent_app, embedder, collection


# ── Load everything ────────────────────────────────────────
try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents ready")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🏢 About")
    st.write("AI-powered HR Policy Assistant. I can answer questions about leave, payroll, and handbook policies directly from company documentation.")
    st.divider()

    st.subheader("📚 Topics Covered")
    for d in DOCUMENTS:
        st.write(f"• {d['topic']}")
    st.divider()

    st.subheader("🔖 Session")
    st.code(f"Thread: {st.session_state.thread_id}")

    if st.session_state.last_meta:
        st.subheader("Last Response Info")
        st.write(f"**Route:** {st.session_state.last_meta.get('route', '—')}")
        faith = st.session_state.last_meta.get("faithfulness", 0)
        color = "🟢" if faith >= 0.7 else "🟡"
        st.write(f"**Faithfulness:** {color} {faith:.2f}")
        sources = st.session_state.last_meta.get("sources", [])
        if sources:
            st.write("**Sources:**")
            for s in sources:
                st.write(f"  • {s}")
    st.divider()

    if st.button("🗑️ New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.last_meta = {}
        st.rerun()

# ============================================================
# CHAT UI
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about PTO, Payroll, or Policies..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Reviewing the handbook..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")

        st.write(answer)

        faith = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        route = result.get("route", "")

        if sources:
            st.caption(f"📎 Sources: {' · '.join(sources)}")
        if faith > 0:
            color = "🟢" if faith >= 0.7 else "🟡"
            st.caption(f"{color} Faithfulness: {faith:.2f}  |  Route: {route}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_meta = {
        "route": route,
        "faithfulness": faith,
        "sources": sources,
    }   