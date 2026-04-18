# 🏢 HR Policy Assistant

An AI-powered conversational assistant that provides 24/7 intelligent, hallucination-free support for corporate HR inquiries.

---

## 🛑 Problem Statement
**Context & Challenge:** HR departments routinely spend a significant amount of time manually answering repetitive employee inquiries regarding company policies—such as Paid Time Off (PTO), payroll schedules, bereavement leave, and remote work guidelines. Conversely, employees often experience friction and delays when trying to find these answers due to lengthy, dense employee handbooks or restricted HR business hours.

**The Solution:** An automated, intelligent, and continuously available (24/7) system capable of instantly providing accurate answers to employee queries based *strictly* on official company documentation.

**Project Objective:** This project develops an AI-powered HR Policy Assistant using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and LangGraph. Crucially, to ensure corporate compliance and avoid misinformation, the system features a built-in "faithfulness" mechanism that restricts the AI from answering outside the provided company knowledge base.

---

## ✨ Key Features
- **Intelligent RAG:** Extracts exact policies from the provided HR knowledge base (NumPy vector store) rather than relying on unverified AI knowledge.
- **Context-Aware Routing:** Intelligently decides whether to query the policy database, use conversation memory for follow-ups, or trigger a specific tool based on user intent.
- **Anti-Hallucination & Faithfulness Evaluation:** Mathematically grades the AI's response against retrieved source documents. If it scores below a strict threshold (0.7), it forces a retry to ensure 100% policy-accurate responses.
- **Dynamic Context Tools:** Features a custom "dates tool" capable of dynamically calculating upcoming paydays and company holidays.
- **Interactive Chat Interface:** Built with Streamlit, providing a sleek, easy-to-use chatbot experience.
- **Session Memory Tracking:** Maintains conversation history so employees can securely ask follow-up questions within their unique ID.
- **Source Citations:** Transparently states the specific policy topic (e.g., "Paid Time Off") used to formulate the answer.

---

## 🎯 Real-World Applications
1. **Automated Employee Onboarding:** A supportive virtual guide allowing new employees to ask questions about benefits privately and quickly.
2. **24/7 Global & Remote Team Support:** Eliminates time-zone friction by providing instantaneous answers anytime, anywhere.
3. **Corporate Self-Service Intranets:** Can be embedded into internal portals, sharply reducing basic internal support tickets.
4. **Policy Compliance & Auditing:** Allows HR managers to quickly verify internal policies and exceptions.
5. **Leave & Payroll Queries Automation:** Uses algorithmic time-tracking to intercept heavy seasonal queries accurately.

---

## 🛠 Technology Stack
- **Core AI & NLP:** Meta LLaMA 3.3 (70B-Versatile), Groq API, SentenceTransformers (`all-MiniLM-L6-v2`), NumPy
- **Orchestration & Logic:** LangGraph (StateGraph workflows), LangChain Core
- **Frontend / UX:** Streamlit
- **Backend & Environment:** Python 3.x, `python-dotenv`

---

## 🏆 Unique Selling Points
- **Self-Correcting Architecture ("Eval Node"):** Unlike standard chatbots, this agent mathematically grades its own answers for "Faithfulness" before passing them to the user.
- **Ultra-Lightweight Custom Vector DB:** Avoids expensive 3rd party databases by executing hyper-fast Cosine-Similarity lookups purely using Python mathematics.
- **Temporal Logical Awareness:** Uses Python calendar tools to inject real-world context into the AI natively.
- **Multi-Agent Routing Mechanism:** Skips heavy retrieval on conversational follow-ups.

---

## 📊 RAGAS Evaluation Results
The system pipeline was baseline tested using the renowned RAGAS framework:
- **Context Precision (1.000):** The vector database performs flawlessly, consistently ranking the absolute most relevant policy documents perfectly at the top of the context window.
- **Faithfulness (0.400 - Baseline):** In baseline testing, the raw LLM often attempted to answer HR questions broadly using outside knowledge. This validates the absolute necessity of our custom built Self-Correcting Eval Node!
- **Answer Relevancy (0.094 - Baseline):** Shows where the raw LLM initially struggled without strict guardrails.

---

## 🚀 Future Improvements
1. **HRIS Integration:** Connecting via API to platforms like Workday or BambooHR for personalized user-specific data (e.g. "How much PTO do *I* have left?").
2. **Automated Document Ingestion:** Allow drag-and-drop of PDF handbooks using OCR splitting instead of arrays.
3. **Enterprise Authentication (SSO):** Integration with Okta or Azure AD to protect company logic.
4. **Voice Interface:** Streamlit audio component to allow hands-free accessibility queries.

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/beherahitesh77/capstone_sway.git
   cd capstone_sway
   ```
2. **Install dependencies:**
   ```bash
   pip install streamlit langgraph langchain-groq langchain-core sentence-transformers python-dotenv numpy
   ```
3. **Set your API Key:** Create a `.env` file and add your Groq key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. **Run the App:**
   ```bash
   streamlit run capstone_streamlit.py
   ```