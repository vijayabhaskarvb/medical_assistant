import os

# =========================
# DISABLE TENSORFLOW FOR TRANSFORMERS
# =========================

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DISABLE_TF"] = "1"

# =========================
# LOAD ENV VARIABLES
# =========================

from dotenv import load_dotenv
load_dotenv()

# =========================
# IMPORT LIBRARIES
# =========================

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# =========================
# EMBEDDING MODEL
# =========================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LOAD PINECONE INDEX
# =========================

docsearch = PineconeVectorStore.from_existing_index(
    index_name="huge-medical-data",
    embedding=embedding
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# =========================
# GROQ LLM
# =========================

chatmodel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# =========================
# PROMPT TEMPLATE (RAG - when docs are found)
# =========================

rag_system_prompt = """
You are an AI medical assistant with access to a medical knowledge database.

Use the retrieved medical context below to answer the user's question accurately and in detail.
If the context supports the answer, use it. If the context is only partially helpful, combine it
with your own medical knowledge to give a complete answer.

Always respond only to medical questions. Structure your answer clearly.

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt),
    ("human", "{input}")
])

# =========================
# FALLBACK PROMPT (when no docs found in Pinecone)
# =========================

FALLBACK_SYSTEM_PROMPT = """
You are a knowledgeable and helpful AI medical assistant.

Answer the user's medical question clearly and in detail using your own medical knowledge.

Structure your response with these sections where relevant:
- Description
- Causes
- Symptoms
- Prevention
- Treatment

IMPORTANT RULES:
- Only answer questions related to medicine, health, diseases, symptoms, medications, anatomy, or medical procedures.
- If the question is NOT related to medicine or health in any way, respond with exactly:
  "I'm a Medical AI Assistant. I can only answer health and medical related questions."
- Do not answer questions about technology, politics, entertainment, general knowledge, or any non-medical topic.
"""

# =========================
# NON-MEDICAL TOPIC GUARD (keywords check for obvious off-topic queries)
# =========================

NON_MEDICAL_KEYWORDS = [
    "cricket", "football", "movie", "film", "actor", "actress", "sport", "game",
    "politics", "election", "president", "prime minister", "stock", "share market",
    "bitcoin", "crypto", "recipe", "cook", "travel", "weather", "news", "celebrity",
    "song", "music", "dance", "exam", "college", "programming", "python", "java",
    "javascript", "code", "software", "iphone", "android", "laptop", "camera"
]

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "how are you"]

# =========================
# CREATE RAG CHAIN
# =========================

qa_chain = create_stuff_documents_chain(chatmodel, rag_prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# =========================
# HELPER: IS ANSWER FROM RAG USEFUL?
# =========================

def is_rag_answer_useful(answer: str) -> bool:
    """
    Check if the RAG chain returned a meaningful answer
    or just said it doesn't have information.
    """
    no_info_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "not present in the context",
        "not in the context",
        "no information in the",
        "context does not",
        "provided context",
        "based on the context, i cannot",
        "the context doesn't",
    ]
    answer_lower = answer.lower()
    return not any(phrase in answer_lower for phrase in no_info_phrases)

# =========================
# CHATBOT FUNCTION
# =========================

def ask_bot(query: str) -> str:

    try:
        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        # ── 1. Handle greetings ──
        if query_lower in GREETINGS:
            return "Hello! I am your Medical AI Assistant. Ask me any health or medical question and I will do my best to help you."

        # ── 2. Quick non-medical keyword guard ──
        for keyword in NON_MEDICAL_KEYWORDS:
            if keyword in query_lower:
                return "I'm a Medical AI Assistant. I can only answer health and medical related questions."

        # ── 3. Try RAG first (search Pinecone vector DB) ──
        docs = retriever.get_relevant_documents(query_stripped)

        if docs:
            # Docs found — run through RAG chain
            rag_response = rag_chain.invoke({"input": query_stripped})
            rag_answer = rag_response.get("answer", "")

            if is_rag_answer_useful(rag_answer):
                # RAG gave a good answer — return it
                return rag_answer

        # ── 4. Fallback — Groq answers from its own knowledge ──
        #       but still enforces medical-only rule via system prompt
        fallback_messages = [
            {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
            {"role": "user",   "content": query_stripped}
        ]

        fallback_response = chatmodel.invoke(fallback_messages)
        return fallback_response.content

    except Exception as e:
        print("CHATBOT ERROR:", e)
        return "Sorry, I couldn't process your medical question right now. Please try again."
