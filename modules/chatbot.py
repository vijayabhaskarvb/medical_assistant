import os
from dotenv import load_dotenv

load_dotenv()

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DISABLE_TF"] = "1"

# All heavy imports are LAZY - loaded only on first use
_rag_chain = None
_retriever = None
_chatmodel = None

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "how are you"]

NON_MEDICAL_KEYWORDS = [
    "cricket", "football", "movie", "film", "actor", "actress", "sport", "game",
    "politics", "election", "president", "prime minister", "stock", "share market",
    "bitcoin", "crypto", "recipe", "cook", "travel", "weather", "news", "celebrity",
    "song", "music", "dance", "exam", "college", "programming", "python", "java",
    "javascript", "code", "software", "iphone", "android", "laptop", "camera"
]

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
"""


def _initialize():
    global _rag_chain, _retriever, _chatmodel

    if _chatmodel is not None:
        return

    print("Initializing chatbot components...")

    from langchain_pinecone import PineconeVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="huge-medical-data",
        embedding=embedding
    )

    _retriever = docsearch.as_retriever(search_kwargs={"k": 3})

    _chatmodel = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    rag_system_prompt = """
You are an AI medical assistant with access to a medical knowledge database.
Use the retrieved medical context below to answer the user's question accurately.
Context:
{context}
"""

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(_chatmodel, rag_prompt)
    _rag_chain = create_retrieval_chain(_retriever, qa_chain)

    print("Chatbot initialized successfully!")


def _is_rag_answer_useful(answer: str) -> bool:
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
    return not any(phrase in answer.lower() for phrase in no_info_phrases)


def ask_bot(query: str) -> str:
    try:
        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        if query_lower in GREETINGS:
            return "Hello! I am your Medical AI Assistant. Ask me any health or medical question."

        for keyword in NON_MEDICAL_KEYWORDS:
            if keyword in query_lower:
                return "I'm a Medical AI Assistant. I can only answer health and medical related questions."

        _initialize()

        docs = _retriever.get_relevant_documents(query_stripped)

        if docs:
            rag_response = _rag_chain.invoke({"input": query_stripped})
            rag_answer = rag_response.get("answer", "")
            if _is_rag_answer_useful(rag_answer):
                return rag_answer

        fallback_messages = [
            {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": query_stripped}
        ]

        fallback_response = _chatmodel.invoke(fallback_messages)
        return fallback_response.content

    except Exception as e:
        print("CHATBOT ERROR:", e)
        return "Sorry, I couldn't process your medical question right now. Please try again."
