"""
=============================================================================
Multi-RAG Q&A Chatbot with Learn & Quiz Modes - A Tutorial Application
=============================================================================

This application demonstrates how to build a RAG (Retrieval-Augmented Generation)
chatbot using LangChain and Streamlit. It's designed as a learning tool for
anyone new to RAG concepts.

WHAT IS RAG?
------------
RAG combines two powerful ideas:
1. RETRIEVAL: Finding relevant documents from a knowledge base
2. GENERATION: Using an LLM to generate answers based on those documents

TWO MODES OF OPERATION:
-----------------------
📖 LEARN MODE: Traditional RAG - you ask questions, the bot answers from documents
🎯 QUIZ MODE:  Role reversal - the bot asks questions, you answer, get graded!

Quiz Mode demonstrates how RAG can be used for:
- Trivia games
- Interview preparation
- Study/flashcard systems
- Knowledge assessment

API SUPPORT:
------------
- OpenAI API (OPENAI_API_KEY)
- OpenRouter API (OPENROUTER_API_KEY) - Access multiple models via one API

Usage:
    streamlit run chatbot.py

Requirements:
    - OPENAI_API_KEY or OPENROUTER_API_KEY in .env file
    - rag_data/ folder with topic JSON files
    - See requirements.txt for Python packages
"""

import os
import glob
import random
import base64
import time
import streamlit as st
from typing import List, Tuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import hashlib
import json

load_dotenv()

# Configuration
RAG_DATA_FOLDER = "rag_data"
LOGO_PATH = "logo.png"
DEFAULT_QUIZ_QUESTIONS = 10

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_api_config() -> Tuple[Optional[str], str, str]:
    """
    Determine which API to use based on available keys.
    Returns: (api_key, base_url, provider_name)
    
    Priority: OPENAI_API_KEY > OPENROUTER_API_KEY
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if openai_key:
        return openai_key, "https://api.openai.com/v1", "OpenAI"
    elif openrouter_key:
        return openrouter_key, OPENROUTER_BASE_URL, "OpenRouter"
    else:
        return None, "", ""


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.7) -> ChatOpenAI:
    """Get LLM instance configured for the active API provider."""
    api_key, base_url, provider = get_api_config()
    
    if provider == "OpenRouter":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            default_headers={"HTTP-Referer": "https://github.com/rag-tutorial"}
        )
    else:
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def get_embeddings() -> OpenAIEmbeddings:
    """Get embeddings instance configured for the active API provider."""
    api_key, base_url, provider = get_api_config()
    
    if provider == "OpenRouter":
        # OpenRouter supports OpenAI embedding models
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
            base_url=base_url
        )
    else:
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

# Page configuration
st.set_page_config(
    page_title="Multi-RAG Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-container {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .title-logo {
        width: 50px;
        height: 50px;
        object-fit: contain;
    }
    .mode-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 10px 0;
    }
    .learn-mode {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .quiz-mode {
        background-color: #fff3e0;
        color: #e65100;
    }
    .quiz-question {
        background-color: #f5f5f5;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    .quiz-feedback-correct {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    .quiz-feedback-partial {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    .quiz-feedback-incorrect {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .score-number {
        font-size: 48px;
        font-weight: bold;
    }
    .out-of-scope {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .progress-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 10px 0;
    }
    .progress-fill {
        background: linear-gradient(90deg, #4caf50, #8bc34a);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


def get_logo_html() -> str:
    """Get logo HTML with base64 encoding or fallback emoji."""
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            return f'<img src="data:image/png;base64,{logo_b64}" class="title-logo"/>'
        except Exception:
            pass
    return '<span style="font-size: 40px;">🤖</span>'


def discover_rag_files() -> List[dict]:
    """
    Discover all RAG JSON files in the data folder.
    
    TUTORIAL NOTE:
    This app supports multiple knowledge bases! Each JSON file in rag_data/
    becomes a separate RAG pipeline. This demonstrates how you can organize
    different topics into separate, switchable knowledge bases.
    """
    rag_files = []
    
    if not os.path.exists(RAG_DATA_FOLDER):
        os.makedirs(RAG_DATA_FOLDER)
        return rag_files
    
    for filepath in glob.glob(os.path.join(RAG_DATA_FOLDER, "*.json")):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                rag_files.append({
                    "filepath": filepath,
                    "name": data.get("name", os.path.basename(filepath)),
                    "description": data.get("description", ""),
                    "icon": data.get("icon", "📄"),
                    "doc_count": len(data.get("documents", [])),
                    "default_mode": data.get("default_mode", "learn"),
                    "quiz_questions": data.get("quiz_questions", DEFAULT_QUIZ_QUESTIONS)
                })
        except Exception as e:
            st.warning(f"Could not load {filepath}: {e}")
    
    return rag_files


def load_rag_documents(filepath: str) -> Tuple[dict, List[dict]]:
    """Load documents from a specific RAG JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data, data.get("documents", [])
    except Exception as e:
        st.error(f"Error loading RAG file: {e}")
        return {}, []


def save_rag_documents(filepath: str, metadata: dict, documents: List[dict]) -> bool:
    """Save documents to a RAG JSON file."""
    try:
        data = {
            "name": metadata.get("name", ""),
            "description": metadata.get("description", ""),
            "icon": metadata.get("icon", "📄"),
            "default_mode": metadata.get("default_mode", "learn"),
            "quiz_questions": metadata.get("quiz_questions", DEFAULT_QUIZ_QUESTIONS),
            "documents": documents
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving RAG file: {e}")
        return False


def generate_doc_id(url: str) -> str:
    """Generate a unique ID for a document based on URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def fetch_url_with_requests(url: str) -> Optional[Tuple[str, List[str]]]:
    """Fetch content and links using requests + BeautifulSoup."""
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        for a_tag in soup.find_all('a', href=True):
            full_url = urljoin(url, a_tag['href'])
            links.append(full_url)
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = '\n'.join(lines)
        
        return (content, links) if len(content) > 100 else None
        
    except Exception:
        return None


def fetch_url_with_playwright(url: str) -> Optional[Tuple[str, List[str]]]:
    """Fetch content and links using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)
            
            links = page.eval_on_selector_all('a[href]', 'elements => elements.map(el => el.href)')
            content = page.inner_text("body")
            browser.close()
            
            return (content, links) if content and len(content) > 100 else None
            
    except Exception as e:
        st.warning(f"Playwright error: {str(e)}")
        return None


def get_local_child_links(base_url: str, links: List[str]) -> List[str]:
    """Filter links within the same folder path (+1 level)."""
    from urllib.parse import urlparse
    
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    base_path = parsed_base.path.rstrip('/')
    
    child_links = set()
    skip_patterns = ['/login', '/signup', '/contact', '/privacy', '/terms', 
                     '/careers', '/press', '.pdf', '.jpg', '.png', '/search']
    
    for link in links:
        try:
            parsed = urlparse(link)
            if parsed.netloc != base_domain:
                continue
            
            link_path = parsed.path.rstrip('/')
            if link_path == base_path:
                continue
            
            if any(p in link_path.lower() for p in skip_patterns):
                continue
            
            if base_path and link_path.startswith(base_path + '/'):
                remaining = link_path[len(base_path):].strip('/')
                if remaining and len(remaining.split('/')) <= 2:
                    child_links.add(link)
            elif not base_path:
                if len(link_path.strip('/').split('/')) <= 2:
                    child_links.add(link)
        except Exception:
            continue
    
    return list(child_links)[:10]


def fetch_url_content(url: str, use_browser: bool = False, crawl_children: bool = False) -> Optional[str]:
    """Fetch content from URL with optional child crawling."""
    all_content = []
    fetched_urls = set()
    
    def fetch_single(target_url):
        if use_browser:
            result = fetch_url_with_playwright(target_url)
            if result:
                return result
        return fetch_url_with_requests(target_url)
    
    result = fetch_single(url)
    if not result:
        st.error("Could not fetch content from URL")
        return None
    
    content, links = result
    all_content.append(f"=== Content from: {url} ===\n{content}")
    fetched_urls.add(url)
    
    if crawl_children and links:
        child_links = get_local_child_links(url, links)
        if child_links:
            st.info(f"Crawling {len(child_links)} child links...")
            for child_url in child_links:
                if child_url not in fetched_urls:
                    child_result = fetch_single(child_url)
                    if child_result:
                        all_content.append(f"\n\n=== Content from: {child_url} ===\n{child_result[0]}")
                        fetched_urls.add(child_url)
    
    return '\n'.join(all_content)


def summarize_content(content: str, url: str) -> Tuple[str, str]:
    """
    Use LLM to generate title and summary for a document.
    
    TUTORIAL NOTE:
    This is an example of using an LLM for a utility task (summarization)
    rather than for the main chat. LLMs are versatile - you can use them
    for extraction, classification, summarization, and more!
    """
    truncated = content[:8000]
    
    prompt = ChatPromptTemplate.from_template(
        """Analyze this web page and provide:
1. A short title (max 6 words)
2. A brief summary (1-2 sentences)

URL: {url}
Content: {content}

Respond exactly as:
TITLE: <title>
SUMMARY: <summary>"""
    )
    
    llm = get_llm(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"url": url, "content": truncated})
        title, summary = "Untitled", "No summary"
        for line in response.strip().split("\n"):
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
        return title, summary
    except Exception:
        return "Untitled Document", f"Document from {url}"


def build_vectorstore(documents: List[dict], collection_name: str = "default") -> Optional[Chroma]:
    """
    Build a vector store from documents - THE HEART OF RAG!
    
    This function performs two critical RAG steps:
    
    1. CHUNKING: Large documents are split into smaller pieces (chunks).
       Why? Because:
       - LLMs have token limits
       - Smaller chunks = more precise retrieval
       - Overlap ensures we don't lose context at boundaries
    
    2. EMBEDDING: Each chunk is converted to a vector (list of numbers).
       These vectors capture the semantic meaning of the text.
       Similar meanings = similar vectors = can be found via similarity search
    
    The vectors are stored in ChromaDB, a vector database optimized for
    fast similarity searches.
    
    IMPORTANT: We use a unique collection_name per topic to prevent
    data from different RAGs mixing together!
    """
    if not documents:
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts, metadatas = [], []
    
    for doc in documents:
        chunks = splitter.split_text(doc.get("content", ""))
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "source": doc.get("source", ""),
                "title": doc.get("title", ""),
                "doc_id": doc.get("id", "")
            })
    
    if not texts:
        return None
    
    # Use unique collection name to prevent cross-contamination between topics
    # Also use ephemeral client (in-memory) to ensure clean state
    embeddings = get_embeddings()
    
    # Create a fresh in-memory Chroma instance with unique collection name
    return Chroma.from_texts(
        texts=texts, 
        metadatas=metadatas, 
        embedding=embeddings,
        collection_name=f"rag_{collection_name}",
        collection_metadata={"hnsw:space": "cosine"}
    )


def check_relevance(question: str, context: str, model: str = "gpt-4o-mini") -> Tuple[bool, float]:
    """
    Check if retrieved context is relevant to the question.
    
    WHY THIS MATTERS:
    RAG retrieval always returns something (the k nearest chunks), but
    "nearest" doesn't mean "relevant". If someone asks about pizza
    in a knowledge base about astronomy, we'll still get astronomy chunks.
    
    This function uses the LLM as a judge to determine if the context
    actually helps answer the question.
    """
    prompt = ChatPromptTemplate.from_template(
        """Determine if the context contains information relevant to answer the question.

Question: {question}

Context: {context}

Respond with only: RELEVANT or NOT_RELEVANT"""
    )
    
    llm = get_llm(model=model, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"question": question, "context": context[:2000]})
        is_relevant = "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()
        return is_relevant, 1.0 if is_relevant else 0.0
    except Exception:
        return True, 0.5


def get_response(question: str, db: Chroma, rag_name: str, conversation_history: List[dict],
                 model: str = "gpt-4o-mini", temperature: float = 0.7) -> Tuple[str, List[dict], bool]:
    """
    Generate a response using the RAG pipeline (LEARN MODE).
    
    This is where the magic happens! The RAG process:
    1. RETRIEVE: Find document chunks most similar to the question
    2. AUGMENT: Build a prompt with the retrieved context
    3. GENERATE: The LLM produces an answer grounded in the context
    """
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [{"source": doc.metadata.get("source", ""), "title": doc.metadata.get("title", "")} for doc in docs]
    
    is_relevant, _ = check_relevance(question, context, model)
    
    if not is_relevant:
        out_of_scope_response = f"""I appreciate your question, but it appears to be outside the scope of the **{rag_name}** knowledge base I'm currently using.

I can only answer questions related to the topics covered in this RAG pipeline. Please try:
- Asking a question related to {rag_name}
- Switching to a different knowledge base using the selector in the sidebar

Would you like to ask something else about {rag_name}?"""
        return out_of_scope_response, [], False
    
    history_text = ""
    if conversation_history:
        recent = conversation_history[-6:]
        history_text = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in recent])
    
    llm = get_llm(model=model, temperature=temperature)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions ONLY based on the provided context from the {rag_name} knowledge base.

### Previous Conversation:
{history}

### Retrieved Context:
{context}

### Current Question:
{question}

### Instructions:
- Answer based ONLY on the retrieved context
- If the context doesn't contain enough information, say so
- Be conversational and helpful
- Use markdown formatting

### Response:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "rag_name": rag_name,
        "history": history_text or "No previous conversation.",
        "context": context,
        "question": question
    })
    
    return response, sources, True


def generate_quiz_question(db: Chroma, rag_name: str, asked_questions: List[str],
                           model: str = "gpt-4o-mini") -> Tuple[str, str, str]:
    """
    Generate a quiz question from the RAG knowledge base.
    
    QUIZ MODE - ROLE REVERSAL:
    Instead of answering questions, the LLM now ASKS questions based on
    the document content. This demonstrates how RAG can be used for:
    - Trivia games
    - Interview preparation  
    - Study/flashcard systems
    
    Returns: (question, correct_answer, source_context)
    """
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Get random content by using a generic query
    random_queries = ["important facts", "key information", "main points", "details", "interesting facts"]
    docs = retriever.invoke(random.choice(random_queries))
    
    if not docs:
        return None, None, None
    
    # Pick a random chunk to base the question on
    doc = random.choice(docs)
    context = doc.page_content
    
    # Format previously asked questions to avoid repetition
    asked_str = "\n".join([f"- {q}" for q in asked_questions[-5:]]) if asked_questions else "None yet"
    
    llm = get_llm(model=model, temperature=0.8)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a quiz master creating questions about {rag_name}.

Based on this content, create ONE interesting quiz question:

Content:
{context}

Previously asked questions (DO NOT repeat these):
{asked_questions}

Requirements:
- Question should be answerable from the content provided
- Make it engaging and specific (not too easy, not impossible)
- Provide a clear, concise correct answer

Respond in this EXACT format:
QUESTION: <your question here>
ANSWER: <the correct answer here>"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "rag_name": rag_name,
            "context": context,
            "asked_questions": asked_str
        })
        
        question, answer = "", ""
        for line in response.strip().split("\n"):
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
        
        if question and answer:
            return question, answer, context
        return None, None, None
        
    except Exception as e:
        st.error(f"Error generating question: {e}")
        return None, None, None


def evaluate_quiz_answer(user_answer: str, correct_answer: str, context: str,
                         question: str, model: str = "gpt-4o-mini") -> Tuple[str, str, int]:
    """
    Evaluate a user's quiz answer against the correct answer.
    
    TUTORIAL NOTE:
    This shows how LLMs can be used as evaluators/judges. Instead of
    exact string matching (which would be too strict), we use the LLM
    to understand if the user's answer captures the correct meaning.
    
    Returns: (feedback_message, feedback_type, points)
    - feedback_type: "correct", "partial", or "incorrect"
    - points: 1 for correct, 0.5 for partial, 0 for incorrect
    """
    llm = get_llm(model=model, temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a fair quiz evaluator. Compare the user's answer to the correct answer.

Question: {question}

Correct Answer: {correct_answer}

User's Answer: {user_answer}

Context (for reference): {context}

Evaluate the user's answer:
- CORRECT: User's answer matches the meaning of the correct answer (exact wording not required)
- PARTIAL: User's answer is partially correct or close but missing key details
- INCORRECT: User's answer is wrong or completely off-topic

Respond in this EXACT format:
VERDICT: <CORRECT, PARTIAL, or INCORRECT>
FEEDBACK: <Brief, encouraging feedback explaining why, and sharing the correct answer if needed>"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "question": question,
            "correct_answer": correct_answer,
            "user_answer": user_answer,
            "context": context[:1500]
        })
        
        verdict, feedback = "INCORRECT", "Let's move on to the next question!"
        for line in response.strip().split("\n"):
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip().upper()
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()
        
        if "CORRECT" in verdict and "INCORRECT" not in verdict and "PARTIAL" not in verdict:
            return feedback, "correct", 1
        elif "PARTIAL" in verdict:
            return feedback, "partial", 0.5
        else:
            return feedback, "incorrect", 0
            
    except Exception:
        return "Let's see the correct answer and move on!", "incorrect", 0


def init_session_state():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "vs_needs_rebuild" not in st.session_state:
        st.session_state.vs_needs_rebuild = True
    if "current_rag" not in st.session_state:
        st.session_state.current_rag = None
    if "current_docs" not in st.session_state:
        st.session_state.current_docs = []
    if "rag_metadata" not in st.session_state:
        st.session_state.rag_metadata = {}
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = "default"
    if "api_validated" not in st.session_state:
        st.session_state.api_validated = False
    
    # Quiz mode state
    if "mode" not in st.session_state:
        st.session_state.mode = "learn"  # "learn" or "quiz"
    if "quiz_active" not in st.session_state:
        st.session_state.quiz_active = False
    if "quiz_questions_asked" not in st.session_state:
        st.session_state.quiz_questions_asked = []
    if "quiz_current_question" not in st.session_state:
        st.session_state.quiz_current_question = None
    if "quiz_current_answer" not in st.session_state:
        st.session_state.quiz_current_answer = None
    if "quiz_current_context" not in st.session_state:
        st.session_state.quiz_current_context = None
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_question_num" not in st.session_state:
        st.session_state.quiz_question_num = 0
    if "quiz_total_questions" not in st.session_state:
        st.session_state.quiz_total_questions = DEFAULT_QUIZ_QUESTIONS
    if "quiz_completed" not in st.session_state:
        st.session_state.quiz_completed = False
    if "quiz_history" not in st.session_state:
        st.session_state.quiz_history = []
    if "awaiting_answer" not in st.session_state:
        st.session_state.awaiting_answer = False


def reset_quiz():
    """Reset quiz state for a new game or when switching topics."""
    st.session_state.quiz_active = False
    st.session_state.quiz_questions_asked = []
    st.session_state.quiz_current_question = None
    st.session_state.quiz_current_answer = None
    st.session_state.quiz_current_context = None
    st.session_state.quiz_score = 0
    st.session_state.quiz_question_num = 0
    st.session_state.quiz_completed = False
    st.session_state.quiz_history = []
    st.session_state.awaiting_answer = False
    st.session_state.pending_question = None  # Also clear any pending learn-mode question


def main():
    init_session_state()
    
    # Check for API keys
    api_key, base_url, provider = get_api_config()
    rag_files = discover_rag_files()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Multi-RAG Chatbot")
        st.markdown("---")
        
        # API Key handling - only show if no key in .env
        if not api_key:
            st.subheader("🔑 API Configuration")
            key_type = st.radio("Provider:", ["OpenAI", "OpenRouter"], horizontal=True)
            entered_key = st.text_input(
                f"{key_type} API Key", 
                type="password",
                placeholder=f"Enter your {key_type} key..."
            )
            if entered_key:
                if key_type == "OpenAI":
                    os.environ["OPENAI_API_KEY"] = entered_key
                else:
                    os.environ["OPENROUTER_API_KEY"] = entered_key
                st.session_state.api_validated = True
                st.rerun()
            st.markdown("---")
        else:
            # Show provider badge (compact, not taking much space)
            st.caption(f"✅ Using **{provider}** API")
        
        # RAG Selector
        st.subheader("📚 Select Knowledge Base")
        
        if not rag_files:
            st.warning("No RAG files found in rag_data/ folder")
            st.stop()
        
        # Random selection on first load
        if st.session_state.current_rag is None:
            random_rag = random.choice(rag_files)
            st.session_state.current_rag = random_rag["filepath"]
            # Set mode based on RAG's default
            st.session_state.mode = random_rag.get("default_mode", "learn")
            st.session_state.quiz_total_questions = random_rag.get("quiz_questions", DEFAULT_QUIZ_QUESTIONS)
            # Set unique collection name
            st.session_state.collection_name = hashlib.md5(random_rag["filepath"].encode()).hexdigest()[:12]
        
        rag_options = {f"{r['icon']} {r['name']} ({r['doc_count']} docs)": r for r in rag_files}
        current_display = None
        for display, rag_info in rag_options.items():
            if rag_info["filepath"] == st.session_state.current_rag:
                current_display = display
                break
        
        selected_display = st.selectbox(
            "Choose a topic:",
            options=list(rag_options.keys()),
            index=list(rag_options.keys()).index(current_display) if current_display else 0
        )
        
        selected_rag = rag_options[selected_display]
        new_rag_path = selected_rag["filepath"]
        
        if new_rag_path != st.session_state.current_rag:
            st.session_state.current_rag = new_rag_path
            st.session_state.vs_needs_rebuild = True
            st.session_state.vectorstore = None  # Clear old vectorstore immediately
            # Create unique collection name from filepath
            st.session_state.collection_name = hashlib.md5(new_rag_path.encode()).hexdigest()[:12]
            st.session_state.messages = []
            st.session_state.mode = selected_rag.get("default_mode", "learn")
            st.session_state.quiz_total_questions = selected_rag.get("quiz_questions", DEFAULT_QUIZ_QUESTIONS)
            reset_quiz()
            st.info("🔄 Switching knowledge base...")
            st.rerun()
        
        # Load current RAG
        rag_meta, docs = load_rag_documents(st.session_state.current_rag)
        st.session_state.current_docs = docs
        st.session_state.rag_metadata = rag_meta
        
        st.caption(rag_meta.get("description", ""))
        
        st.markdown("---")
        
        # Mode Toggle
        st.subheader("🎮 Mode")
        mode_options = {"📖 Learn": "learn", "🎯 Quiz": "quiz"}
        current_mode_display = "📖 Learn" if st.session_state.mode == "learn" else "🎯 Quiz"
        
        selected_mode = st.radio(
            "Select mode:",
            options=list(mode_options.keys()),
            index=0 if st.session_state.mode == "learn" else 1,
            horizontal=True
        )
        
        new_mode = mode_options[selected_mode]
        if new_mode != st.session_state.mode:
            st.session_state.mode = new_mode
            st.session_state.messages = []
            reset_quiz()
            st.rerun()
        
        if st.session_state.mode == "learn":
            st.info("💡 **Learn Mode**: Ask questions and get answers from the knowledge base.")
        else:
            st.info("🎯 **Quiz Mode**: Test your knowledge! The bot asks, you answer.")
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        if st.session_state.mode == "learn":
            show_sources = st.checkbox("Show Sources", value=True)
        else:
            show_sources = False
        
        use_browser = st.checkbox("Use Browser (Playwright)", value=False)
        crawl_children = st.checkbox("Crawl Child Links", value=True)
        
        st.markdown("---")
        
        # Add Documents
        with st.expander("➕ Add Documents", expanded=False):
            st.caption("Add URLs to current knowledge base")
            urls_input = st.text_area("URLs (one per line)", height=80, label_visibility="collapsed",
                                      placeholder="https://example.com/page1")
            
            if st.button("🔍 Fetch & Add", use_container_width=True):
                if not get_api_config()[0]:
                    st.error("API key required")
                elif urls_input.strip():
                    urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()]
                    if urls:
                        added = 0
                        
                        for url in urls:
                            doc_id = generate_doc_id(url)
                            if any(d["id"] == doc_id for d in st.session_state.current_docs):
                                st.warning(f"Already exists: {url[:40]}...")
                                continue
                            
                            content = fetch_url_content(url, use_browser, crawl_children)
                            if content and len(content) > 100:
                                title, summary = summarize_content(content, url)
                                st.session_state.current_docs.append({
                                    "id": doc_id, "title": title, "source": url,
                                    "content": content, "summary": summary
                                })
                                added += 1
                        
                        if added > 0:
                            save_rag_documents(st.session_state.current_rag, 
                                             st.session_state.rag_metadata,
                                             st.session_state.current_docs)
                            st.session_state.vs_needs_rebuild = True
                            st.success(f"✓ Added {added} document(s)")
                            st.rerun()
        
        # Document List
        st.markdown(f"**Documents ({len(st.session_state.current_docs)}):**")
        for doc in st.session_state.current_docs:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{doc['title'][:30]}**")
            with col2:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Remove"):
                    st.session_state.current_docs = [d for d in st.session_state.current_docs if d["id"] != doc["id"]]
                    save_rag_documents(st.session_state.current_rag, st.session_state.rag_metadata, st.session_state.current_docs)
                    st.session_state.vs_needs_rebuild = True
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat / Reset Quiz", use_container_width=True):
            st.session_state.messages = []
            reset_quiz()
            st.rerun()
        
        st.caption("Built with LangChain + Streamlit")
    
    # Main Content
    logo_html = get_logo_html()
    rag_icon = st.session_state.rag_metadata.get("icon", "📄")
    rag_name = st.session_state.rag_metadata.get("name", "Knowledge Base")
    
    st.markdown(f"""
    <div class="title-container">
        {logo_html}
        <h1 style="margin: 0;">RAG Q&A Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode badge
    mode_class = "learn-mode" if st.session_state.mode == "learn" else "quiz-mode"
    mode_text = "📖 LEARN MODE" if st.session_state.mode == "learn" else "🎯 QUIZ MODE"
    st.markdown(f'<span class="mode-badge {mode_class}">{mode_text}</span>', unsafe_allow_html=True)
    
    st.caption(f"Currently using: {rag_icon} **{rag_name}** • Switch topics or modes in the sidebar")
    
    if not get_api_config()[0]:
        st.warning("⚠️ Enter your API key (OpenAI or OpenRouter) in the sidebar")
        st.stop()
    
    if not st.session_state.current_docs:
        st.warning("⚠️ No documents in this knowledge base. Add some URLs!")
        st.stop()
    
    # Build vectorstore
    if st.session_state.vs_needs_rebuild:
        with st.spinner(f"🔧 Loading {rag_name}... Creating embeddings and building vector index..."):
            st.session_state.vectorstore = build_vectorstore(
                st.session_state.current_docs,
                st.session_state.collection_name
            )
            st.session_state.vs_needs_rebuild = False
        st.success(f"✅ {rag_name} ready! ({len(st.session_state.current_docs)} documents loaded)")
        time.sleep(1)
        st.rerun()
    
    if not st.session_state.vectorstore:
        st.error("Failed to build vector store")
        st.stop()
    
    # ==========================================
    # LEARN MODE
    # ==========================================
    if st.session_state.mode == "learn":
        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("out_of_scope"):
                    st.markdown(f'<div class="out-of-scope">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])
                
                if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                    with st.expander("📎 Sources"):
                        seen = set()
                        for src in msg["sources"]:
                            if src.get("source") and src["source"] not in seen:
                                st.markdown(f"- [{src.get('title', 'Source')}]({src['source']})")
                                seen.add(src["source"])
        
        # Chat input
        if prompt := st.chat_input(f"Ask about {rag_name}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources, is_relevant = get_response(
                        prompt, st.session_state.vectorstore, rag_name,
                        st.session_state.messages[:-1], model, temperature
                    )
                    
                    if not is_relevant:
                        st.markdown(f'<div class="out-of-scope">{response}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(response)
                        if show_sources and sources:
                            with st.expander("📎 Sources"):
                                seen = set()
                                for src in sources:
                                    if src.get("source") and src["source"] not in seen:
                                        st.markdown(f"- [{src.get('title', 'Source')}]({src['source']})")
                                        seen.add(src["source"])
                    
                    st.session_state.messages.append({
                        "role": "assistant", "content": response,
                        "sources": sources, "out_of_scope": not is_relevant
                    })
        
        # Sample questions (when chat is empty)
        if not st.session_state.messages:
            st.markdown("### 💡 Try asking:")
            
            sample_qs = {
                "prompting": ["What is chain-of-thought prompting?", "How do I structure a good prompt?"],
                "atlas": ["What is 3I/ATLAS?", "When was the comet discovered?"],
                "vegan": ["What are good vegan protein sources?", "How do I make tofu scramble?"],
                "travel": ["How far ahead should I book flights?", "What are budget travel tips?"],
                "movie": ["What's the trivia about Jaws?", "Tell me about The Godfather making"],
                "animal": ["How do octopuses escape?", "How do dolphins sleep?"]
            }
            
            rag_key = None
            rag_lower = rag_name.lower()
            for key in sample_qs:
                if key in rag_lower:
                    rag_key = key
                    break
            
            questions = sample_qs.get(rag_key, ["What topics are covered here?", "Tell me something interesting"])
            
            cols = st.columns(2)
            for i, q in enumerate(questions[:4]):
                with cols[i % 2]:
                    if st.button(q, key=f"sample_{i}", use_container_width=True):
                        st.session_state.pending_question = q
                        st.rerun()
        
        # Process pending question
        if st.session_state.pending_question:
            pending = st.session_state.pending_question
            st.session_state.pending_question = None
            
            st.session_state.messages.append({"role": "user", "content": pending})
            
            with st.chat_message("user"):
                st.markdown(pending)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources, is_relevant = get_response(
                        pending, st.session_state.vectorstore, rag_name,
                        st.session_state.messages[:-1], model, temperature
                    )
                    
                    if not is_relevant:
                        st.markdown(f'<div class="out-of-scope">{response}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(response)
                    
                    st.session_state.messages.append({
                        "role": "assistant", "content": response,
                        "sources": sources, "out_of_scope": not is_relevant
                    })
    
    # ==========================================
    # QUIZ MODE
    # ==========================================
    else:
        total_q = st.session_state.quiz_total_questions
        
        # Quiz completed - show final score
        if st.session_state.quiz_completed:
            score = st.session_state.quiz_score
            percentage = (score / total_q) * 100
            
            st.markdown(f"""
            <div class="score-card">
                <h2>🎉 Quiz Complete!</h2>
                <div class="score-number">{score}/{total_q}</div>
                <p style="font-size: 24px;">{percentage:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance message
            if percentage >= 80:
                st.success("🌟 Excellent! You really know your stuff!")
            elif percentage >= 60:
                st.info("👍 Good job! Keep learning!")
            else:
                st.warning("📚 Keep studying! You'll get better!")
            
            # Show quiz history
            with st.expander("📋 Review Your Answers", expanded=False):
                for i, item in enumerate(st.session_state.quiz_history, 1):
                    result_emoji = "✅" if item["result"] == "correct" else ("🟡" if item["result"] == "partial" else "❌")
                    st.markdown(f"**Q{i}: {item['question']}**")
                    st.markdown(f"Your answer: {item['user_answer']}")
                    st.markdown(f"Correct answer: {item['correct_answer']}")
                    st.markdown(f"Result: {result_emoji} {item['feedback']}")
                    st.markdown("---")
            
            if st.button("🔄 Play Again", use_container_width=True):
                reset_quiz()
                st.rerun()
        
        # Quiz not started yet
        elif not st.session_state.quiz_active:
            st.markdown(f"""
            ### 🎯 Ready to test your knowledge about {rag_name}?
            
            - **{total_q} questions** will be asked
            - Answer each question to the best of your ability
            - You'll get feedback after each answer
            - Final score shown at the end
            
            *Tip: The questions are generated from the knowledge base, so they're always relevant!*
            """)
            
            # Safety check: don't allow quiz start if vectorstore isn't ready
            if st.session_state.vectorstore is None or st.session_state.vs_needs_rebuild:
                st.warning("⏳ Please wait for the knowledge base to finish loading...")
            elif st.button("🚀 Start Quiz", use_container_width=True, type="primary"):
                st.session_state.quiz_active = True
                st.session_state.quiz_question_num = 1
                # Generate first question
                with st.spinner("Generating your first question..."):
                    q, a, ctx = generate_quiz_question(
                        st.session_state.vectorstore, rag_name,
                        st.session_state.quiz_questions_asked, model
                    )
                    if q and a:
                        st.session_state.quiz_current_question = q
                        st.session_state.quiz_current_answer = a
                        st.session_state.quiz_current_context = ctx
                        st.session_state.quiz_questions_asked.append(q)
                        st.session_state.awaiting_answer = True
                st.rerun()
        
        # Quiz in progress
        else:
            # Progress bar
            progress = st.session_state.quiz_question_num / total_q
            st.markdown(f"**Question {st.session_state.quiz_question_num} of {total_q}** | Score: {st.session_state.quiz_score}")
            st.markdown(f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress * 100}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display quiz history
            for item in st.session_state.quiz_history:
                st.markdown(f'<div class="quiz-question">❓ {item["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**Your answer:** {item['user_answer']}")
                
                feedback_class = f"quiz-feedback-{item['result']}"
                result_emoji = "✅" if item["result"] == "correct" else ("🟡" if item["result"] == "partial" else "❌")
                st.markdown(f'<div class="{feedback_class}">{result_emoji} {item["feedback"]}</div>', unsafe_allow_html=True)
                st.markdown("")
            
            # Current question
            if st.session_state.quiz_current_question and st.session_state.awaiting_answer:
                st.markdown(f'<div class="quiz-question">❓ {st.session_state.quiz_current_question}</div>', unsafe_allow_html=True)
                
                # Answer input
                user_answer = st.text_input("Your answer:", key=f"quiz_answer_{st.session_state.quiz_question_num}",
                                           placeholder="Type your answer here...")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Submit Answer", use_container_width=True, type="primary", disabled=not user_answer):
                        if user_answer:
                            with st.spinner("Evaluating your answer..."):
                                feedback, result, points = evaluate_quiz_answer(
                                    user_answer,
                                    st.session_state.quiz_current_answer,
                                    st.session_state.quiz_current_context,
                                    st.session_state.quiz_current_question,
                                    model
                                )
                            
                            # Record this Q&A
                            st.session_state.quiz_history.append({
                                "question": st.session_state.quiz_current_question,
                                "user_answer": user_answer,
                                "correct_answer": st.session_state.quiz_current_answer,
                                "feedback": feedback,
                                "result": result,
                                "points": points
                            })
                            
                            st.session_state.quiz_score += points
                            st.session_state.awaiting_answer = False
                            
                            # Check if quiz is done
                            if st.session_state.quiz_question_num >= total_q:
                                st.session_state.quiz_completed = True
                            
                            st.rerun()
                
                with col2:
                    if st.button("Skip Question", use_container_width=True):
                        # Record as skipped
                        st.session_state.quiz_history.append({
                            "question": st.session_state.quiz_current_question,
                            "user_answer": "(skipped)",
                            "correct_answer": st.session_state.quiz_current_answer,
                            "feedback": f"The answer was: {st.session_state.quiz_current_answer}",
                            "result": "incorrect",
                            "points": 0
                        })
                        st.session_state.awaiting_answer = False
                        
                        if st.session_state.quiz_question_num >= total_q:
                            st.session_state.quiz_completed = True
                        
                        st.rerun()
            
            # Generate next question button (after answering)
            elif not st.session_state.awaiting_answer and not st.session_state.quiz_completed:
                if st.button("➡️ Next Question", use_container_width=True, type="primary"):
                    st.session_state.quiz_question_num += 1
                    with st.spinner("Generating next question..."):
                        q, a, ctx = generate_quiz_question(
                            st.session_state.vectorstore, rag_name,
                            st.session_state.quiz_questions_asked, model
                        )
                        if q and a:
                            st.session_state.quiz_current_question = q
                            st.session_state.quiz_current_answer = a
                            st.session_state.quiz_current_context = ctx
                            st.session_state.quiz_questions_asked.append(q)
                            st.session_state.awaiting_answer = True
                        else:
                            # If we can't generate more questions, end the quiz
                            st.session_state.quiz_completed = True
                    st.rerun()


if __name__ == "__main__":
    main()