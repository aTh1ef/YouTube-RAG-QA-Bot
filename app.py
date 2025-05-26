# app.py - Enhanced YouTube Q&A Bot with Fact Checking
import streamlit as st
import os
import requests
import re
import logging
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.parse import quote_plus
import hashlib
import wikipedia
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from urllib.parse import urljoin
import warnings
import requests  # We'll use requests instead of serpapi
warnings.filterwarnings("ignore")

# Data classes
@dataclass
class Claim:
    text: str
    source_chunk: str
    timestamp: str = ""
    category: str = ""

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    source_domain: str

def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return ""

def search_sources_for_claim(claim: str, max_results: int = 10) -> List[SearchResult]:
    """
    Search for sources related to a claim using Google Custom Search.
    Returns a simple list of relevant sources without making judgments.
    """
    results = []
    
    try:
        params = {
            'key': GOOGLE_SEARCH_API_KEY,
            'cx': GOOGLE_SEARCH_ENGINE_ID,
            'q': claim,
            'num': min(max_results, 10),
            'safe': 'active'
        }
        
        response = requests.get(
            'https://www.googleapis.com/customsearch/v1',
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            for item in items:
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                link = item.get('link', '')
                domain = extract_domain(link)
                
                if all([title, snippet, link]):
                    result = SearchResult(
                        title=title,
                        snippet=snippet,
                        url=link,
                        source_domain=domain
                    )
                    results.append(result)
                
    except Exception as e:
        logger.error(f"Google Custom Search error: {str(e)}")
        st.error(f"Error searching for sources: {str(e)}")
    
    return results

def render_sources_ui(sources: List[SearchResult]):
    """
    Render the list of sources in the UI.
    """
    if not sources:
        st.info("No sources found for this claim.")
        return
        
    st.markdown("### 🔍 Related Sources")
    
    for i, source in enumerate(sources, 1):
        with st.container():
            st.markdown(f"**{i}. [{source.title}]({source.url})**")
            st.markdown(f"📍 *{source.source_domain}*")
            st.markdown(f"{source.snippet}")
            st.markdown("---")

def render_individual_claims_ui(claims: List[Claim]):
    """Render UI for showing claims and their sources."""
    if not claims:
        st.info("No claims available to check.")
        return
    
    # Display each claim with a search button
    for i, claim in enumerate(claims):
        with st.expander(f"Claim {i+1}: {claim.text[:100]}{'...' if len(claim.text) > 100 else ''}", expanded=False):
            # Display claim details
            st.markdown(f"**Full Claim:** {claim.text}")
            st.markdown(f"**Source Chunk:** {claim.source_chunk}")
            
            # Search button
            if st.button(f"🔍 Find Sources for Claim {i+1}", key=f"search_{i}"):
                with st.spinner("Searching for sources..."):
                    sources = search_sources_for_claim(claim.text)
                    render_sources_ui(sources)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("youtube_qa_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("youtube_qa_bot")

# LangChain imports
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import StdOutCallbackHandler

# Initialize Pinecone with the updated client
import pinecone
from langchain_pinecone import PineconeVectorStore

# YouTube Transcript API
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

# Data classes for fact-checking
@dataclass
class FactCheckResult:
    claim: Claim
    verdict: str  # "TRUE", "FALSE", "UNCERTAIN"
    confidence: float
    explanation: str
    sources: List[SearchResult]
    evidence_summary: str

# Set page configuration
st.set_page_config(
    page_title="YouTube Video Q&A Bot with Fact Checker",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state variables
session_vars = [
    'transcription', 'video_title', 'video_url', 'vector_store', 
    'video_namespace', 'conversation', 'chat_history', 'extracted_claims',
    'fact_check_results', 'fact_check_in_progress'
]

for var in session_vars:
    if var not in st.session_state:
        if var in ['chat_history', 'extracted_claims', 'fact_check_results']:
            st.session_state[var] = []
        elif var == 'fact_check_in_progress':
            st.session_state[var] = False
        else:
            st.session_state[var] = None

# Load API keys from Streamlit secrets
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT", "gcp-starter")
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]  # Make it required, not optional!
    GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
    GOOGLE_SEARCH_ENGINE_ID = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]

    # Initialize Pinecone client
    pinecone_version = pinecone.__version__.split('.')[0]
    if int(pinecone_version) >= 4:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    else:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        pc = pinecone
except Exception as e:
    st.error(f"Error loading API keys: {str(e)}")
    st.error("Please make sure you have set up your .streamlit/secrets.toml file with all required API keys.")
    st.stop()

# Configuration
PINECONE_INDEX_NAME = "youtube-qa-bot"
# Enhanced credible domains with better scoring
CREDIBLE_DOMAINS = {
    # News & Media (High Priority)
    'reuters.com': 0.95, 'ap.org': 0.95, 'bbc.com': 0.90,
    'npr.org': 0.90, 'pbs.org': 0.85, 'wsj.com': 0.85,
    'nytimes.com': 0.85, 'washingtonpost.com': 0.85,
    'theguardian.com': 0.80, 'economist.com': 0.85,
    
    # Academic & Research (Highest Priority)
    'edu': 0.95, 'gov': 0.98, 'nature.com': 0.95,
    'science.org': 0.95, 'cell.com': 0.95,
    'pubmed.ncbi.nlm.nih.gov': 0.95, 'nih.gov': 0.95,
    'who.int': 0.95, 'cdc.gov': 0.95,
    
    # Reference & Encyclopedia
    'britannica.com': 0.85, 'merriam-webster.com': 0.80,
    'wikipedia.org': 0.70,  # Reduced Wikipedia credibility
    
    # Tech & Business (Specialized)
    'techcrunch.com': 0.75, 'arstechnica.com': 0.80,
    'wired.com': 0.75, 'forbes.com': 0.75,
    'bloomberg.com': 0.85, 'cnbc.com': 0.75,
    
    # Fact-checking sites (High Priority)
    'snopes.com': 0.90, 'factcheck.org': 0.95,
    'politifact.com': 0.90, 'fullfact.org': 0.90
}

# Cache models to avoid reinitialization
@st.cache_resource
def get_embeddings_model():
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=None
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {str(e)}")
        st.error(f"Failed to initialize embeddings model: {str(e)}")
        return None

@st.cache_resource
def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2048
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.0-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=2048
            )
        except Exception as e2:
            logger.error(f"Error initializing fallback LLM: {str(e2)}")
            st.error(f"Failed to initialize language model: {str(e2)}")
            return None

# Existing functions (extract_video_id, get_video_title, get_youtube_transcript, etc.)
def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/e\/|youtube\.com\/user\/.+\/|youtube\.com\/user\/(?!.+\/)|youtube\.com\/.*[?&]v=|youtube\.com\/.*[?&]vi=)([^#&?\/\s]{11})',
        r'(?:youtube\.com\/shorts\/|youtube\.com\/live\/)([^#&?\/\s]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_title(video_id):
    try:
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}")
        if response.status_code == 200:
            match = re.search(r'<title>(.+?) - YouTube</title>', response.text)
            if match:
                return match.group(1)
        return f"YouTube Video {video_id}"
    except Exception as e:
        logger.warning(f"Could not get video title: {str(e)}")
        return f"YouTube Video {video_id}"

def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript_list:
            st.error("No transcript content available for this video.")
            return None
        full_transcript = " ".join([part["text"] + " " for part in transcript_list])
        if not full_transcript.strip():
            st.error("Transcript is empty after processing.")
            return None
        return full_transcript
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"No transcript available for this video: {str(e)}")
        return None
        
def polish_transcript_with_gemini(raw_transcript: str, video_title: str) -> str:
    """
    Polish raw YouTube transcript using Gemini to make it context-aware and AI-friendly.
    """
    llm = get_llm()
    if not llm:
        logger.warning("LLM not available for transcript polishing, using raw transcript")
        return raw_transcript
    
    polishing_prompt = """
You are an expert transcript editor. Your task is to transform a raw YouTube transcript into a well-structured, context-aware document that preserves ALL original meaning while making it readable and AI-friendly.

VIDEO TITLE: {video_title}

INSTRUCTIONS:
🔤 1. Structure the Text into Complete Paragraphs
- Each paragraph should focus on a single idea or topic
- Avoid mixing unrelated points in the same paragraph  
- Ensure each paragraph is self-contained and doesn't rely on previous ones to be understood

🔍 2. Clarify Ambiguity
- Replace vague references like "this," "it," "they," "that thing" with the actual subject or entity
- Use the video title and context to identify what specific products, people, or concepts are being discussed
- Ensure the reader can understand the meaning without having to guess what is being discussed

✍️ 3. Correct Language Issues
- Fix grammar, punctuation, and spelling
- Remove filler words like "um," "uh," "you know," "like," etc.
- Turn casual spoken phrases into readable written language
- Fix incomplete sentences and run-on sentences

🧠 4. Preserve All Meaning
- Do NOT summarize or shorten the transcript
- Keep all opinions, data, references, comparisons, and examples intact
- Keep all numbers, statistics, and factual claims exactly as stated
- Your goal is clarity and structure, not brevity

🤖 5. Make It AI-Friendly
- Ensure each paragraph can be processed independently for NLP tasks
- Avoid splitting sentences or ideas across multiple paragraphs
- Replace unclear pronouns with specific nouns
- Clarify technical terms and product names when first mentioned

IMPORTANT: This is for a fact-checking system, so accuracy and clarity are crucial. Every claim and detail must be preserved.

RAW TRANSCRIPT:
{raw_transcript}

POLISHED TRANSCRIPT:
"""

    try:
        # Split large transcripts into smaller chunks to avoid token limits
        max_chunk_size = 10000  # Adjust based on Gemini's context window
        
        if len(raw_transcript) <= max_chunk_size:
            # Process the entire transcript at once
            prompt = polishing_prompt.format(
                video_title=video_title,
                raw_transcript=raw_transcript
            )
            
            with st.spinner("🔧 Polishing transcript with AI for better context awareness..."):
                response = llm.invoke(prompt)
                polished_transcript = response.content.strip()
                
                if polished_transcript and len(polished_transcript) > 100:
                    logger.info(f"Successfully polished transcript: {len(raw_transcript)} -> {len(polished_transcript)} characters")
                    return polished_transcript
                else:
                    logger.warning("Polished transcript too short, using original")
                    return raw_transcript
        else:
            # Process in chunks for very long transcripts
            chunks = []
            words = raw_transcript.split()
            current_chunk = []
            current_size = 0
            
            for word in words:
                current_chunk.append(word)
                current_size += len(word) + 1
                
                if current_size >= max_chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            polished_chunks = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunks):
                status_text.text(f"🔧 Polishing transcript chunk {i+1} of {len(chunks)}...")
                
                chunk_prompt = polishing_prompt.format(
                    video_title=video_title,
                    raw_transcript=chunk
                )
                
                try:
                    response = llm.invoke(chunk_prompt)
                    polished_chunk = response.content.strip()
                    
                    if polished_chunk and len(polished_chunk) > 50:
                        polished_chunks.append(polished_chunk)
                    else:
                        polished_chunks.append(chunk)  # Fallback to original
                        
                except Exception as e:
                    logger.warning(f"Error polishing chunk {i}: {str(e)}")
                    polished_chunks.append(chunk)  # Fallback to original
                
                progress_bar.progress((i + 1) / len(chunks))
                time.sleep(0.5)  # Rate limiting
            
            progress_bar.empty()
            status_text.empty()
            
            polished_transcript = "\n\n".join(polished_chunks)
            logger.info(f"Successfully polished transcript in {len(chunks)} chunks: {len(raw_transcript)} -> {len(polished_transcript)} characters")
            
            return polished_transcript
            
    except Exception as e:
        logger.error(f"Error polishing transcript: {str(e)}")
        st.warning(f"Could not polish transcript: {str(e)}. Using original transcript.")
        return raw_transcript
        
def split_text_into_documents(text, video_title):
    try:
        doc = Document(page_content=text, metadata={"source": video_title})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        docs = text_splitter.split_documents([doc])
        for i, chunk in enumerate(docs):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["video_title"] = video_title
        return docs
    except Exception as e:
        logger.error(f"Error splitting text into documents: {str(e)}")
        st.error(f"Error splitting text into documents: {str(e)}")
        return []

def create_and_store_embeddings(docs, video_title):
    try:
        video_namespace = ''.join(e for e in video_title if e.isalnum()).lower()[:40]
        video_namespace = video_namespace.replace(" ", "-")
        
        embeddings = get_embeddings_model()
        if embeddings is None:
            st.error("Failed to initialize embeddings model. Cannot continue.")
            return None, None

        # Handle Pinecone index creation/management
        try:
            pinecone_version = pinecone.__version__.split('.')[0]
            if int(pinecone_version) >= 4:
                index_names = [idx["name"] for idx in pc.list_indexes()]
                if PINECONE_INDEX_NAME not in index_names:
                    st.warning(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new index...")
                    pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=384,
                        metric="cosine"
                    )
                    st.success(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
            else:
                index_names = pc.list_indexes()
                if PINECONE_INDEX_NAME not in index_names:
                    st.warning(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new index...")
                    pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=384,
                        metric="cosine"
                    )
                    st.success(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.error(f"Error checking Pinecone indexes: {str(e)}")
            st.error(f"Error checking Pinecone indexes: {str(e)}")
            return None, None

        with st.spinner("Creating and storing embeddings..."):
            try:
                pinecone_version = pinecone.__version__.split('.')[0]
                if int(pinecone_version) >= 4:
                    index = pc.Index(PINECONE_INDEX_NAME)
                    try:
                        index.delete(namespace=video_namespace, delete_all=True)
                        logger.info(f"Deleted existing vectors in namespace: {video_namespace}")
                    except Exception as e:
                        logger.warning(f"No existing vectors to delete in namespace {video_namespace}: {str(e)}")
                else:
                    index = pc.Index(PINECONE_INDEX_NAME)
                    try:
                        index.delete(deleteAll=True, namespace=video_namespace)
                        logger.info(f"Deleted existing vectors in namespace: {video_namespace}")
                    except Exception as e:
                        logger.warning(f"No existing vectors to delete in namespace {video_namespace}: {str(e)}")

                vector_store = PineconeVectorStore.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX_NAME,
                    namespace=video_namespace
                )
                logger.info(f"Successfully stored {len(docs)} document chunks in Pinecone")
            except Exception as e:
                logger.error(f"Error storing embeddings: {str(e)}")
                st.error(f"Error storing embeddings: {str(e)}")
                return None, None

        return vector_store, video_namespace
    except Exception as e:
        logger.error(f"Error in create_and_store_embeddings: {str(e)}")
        st.error(f"Error storing embeddings: {str(e)}")
        return None, None

def setup_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

def setup_qa_chain(vector_store):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = get_llm()
        if llm is None:
            st.error("Failed to initialize language model. Cannot continue.")
            return None
        memory = setup_memory()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        st.error(f"Error setting up QA chain: {str(e)}")
        return None

def process_query(query, qa_chain):
    try:
        with st.spinner("Searching for answer..."):
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "Sorry, I couldn't find an answer in the transcript.")
            source_docs = result.get("source_documents", [])
            logger.info(f"Retrieved {len(source_docs)} source documents for query: {query}")
            if source_docs:
                for i, doc in enumerate(source_docs[:2]):
                    logger.info(f"Source doc {i}: {doc.page_content[:100]}...")
            else:
                logger.warning("No source documents retrieved!")
                if "I don't know" in answer or "I couldn't find" in answer or not answer.strip():
                    answer = "I couldn't find specific information about that in the video transcript. Could you try rephrasing your question or asking about another aspect of the video?"
            return answer, source_docs
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error(f"Error processing query: {str(e)}")
        return f"I encountered an error while processing your query: {str(e)}", []

# NEW FACT-CHECKING FUNCTIONS

def extract_claims_from_transcript(transcript_chunks: List[Document]) -> List[Claim]:
    """Extract factual claims from transcript chunks using Gemini with improved specificity."""
    llm = get_llm()
    if not llm:
        return []
    
    claims = []
    
    # IMPROVED CLAIM EXTRACTION PROMPT
    claim_extraction_prompt = """
Analyze the following transcript chunk and extract ONLY specific, factual claims that can be independently verified.

🟢 INCLUDE these types of verifiable claims:
✅ **Numbers and statistics**: "The population is over 1.4 billion", "Revenue rose by 25% in 2023"
✅ **Technical specifications**: "The phone has 8GB RAM", "Uses a 3nm chip", "It supports 5G connectivity"
✅ **Scientific facts or assertions**: "Water boils at 100°C", "The human brain has over 80 billion neurons"
✅ **Historical facts/events**: "World War II ended in 1945", "Tesla was founded in 2003"
✅ **Performance claims**: "Battery lasts 12 hours", "Processes data 30% faster than the old model"
✅ **Product feature assertions**: "Includes IP68 waterproofing", "Supports wireless charging"
✅ **Economic/financial statements**: "Inflation hit 7% last year", "Bitcoin surged to $65,000 in 2021"
✅ **Geopolitical claims**: "India is the world's largest democracy", "NATO was formed in 1949"
✅ **Medical/health-related facts**: "Vitamin C boosts immunity", "This treatment reduces mortality by 20%"
✅ **Comparisons with specifics**: "50% faster than previous model", "Costs $200 less than competitor"
✅ **Named entity facts**: "Barack Obama was the 44th U.S. President", "The Amazon is the largest rainforest"

🔴 EXCLUDE the following:
❌ **Just names**: "iPhone 15", "Elon Musk", "NVIDIA"
❌ **Vague references**: "This thing is amazing", "They say it works well"
❌ **Subjective opinions**: "I think it's beautiful", "Sounds awesome", "In my opinion..."
❌ **Generalized praise or criticism**: "Great product", "Terrible decision", "Amazing build quality"
❌ **Advice or recommendations**: "You should try it", "I recommend buying it"
❌ **Questions or hypotheticals**: "Should you invest?", "What if it fails?"
❌ **Speculation or predictions**: "It might succeed", "Could change the world"
❌ **Personal experiences**: "When I used it...", "I noticed..."

📌 IMPORTANT RULES:
- Each claim must be a **complete, standalone factual statement**
- It must contain **specific, verifiable information** (numbers, names, data, relationships)
- The claim must **make sense without any external or surrounding context**
- Only extract **objective facts** that can be verified against reliable sources

📦 Output Format:
For each valid claim, return a JSON object with:
- "claim": the complete factual statement (self-contained)
- "category": the category of the claim (e.g., technical_spec, historical, performance, scientific, economic, geopolitical, health, comparison)

If there are no valid claims, return an empty JSON array: `[]`

Transcript chunk:
{chunk_content}

Response format: JSON array of claim objects only
"""

    
    for i, chunk in enumerate(transcript_chunks):
        try:
            prompt = claim_extraction_prompt.format(chunk_content=chunk.page_content)
            response = llm.invoke(prompt)
            
            # Parse JSON response
            try:
                # Clean the response to extract JSON
                response_text = response.content.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                claims_data = json.loads(response_text)
                
                if isinstance(claims_data, list):
                    for claim_data in claims_data:
                        if isinstance(claim_data, dict) and 'claim' in claim_data:
                            claim = Claim(
                                text=claim_data['claim'],
                                source_chunk=chunk.page_content,
                                category=claim_data.get('category', 'general')
                            )
                            claims.append(claim)
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON from claim extraction response: {e}")
                # Fallback: try to extract claims from plain text response
                if "claim" in response.content.lower():
                    lines = response.content.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            claim = Claim(
                                text=line.strip(),
                                source_chunk=chunk.page_content,
                                category='general'
                            )
                            claims.append(claim)
                            
        except Exception as e:
            logger.error(f"Error extracting claims from chunk {i}: {str(e)}")
            continue
    
    return claims

def search_with_serpapi(query: str, max_results: int = 8) -> List[SearchResult]:
    """
    Enhanced SerpAPI search with better error handling and more results.
    """
    results = []
    
    if not SERPAPI_KEY:
        logger.error("SerpAPI key not found!")
        return results
        
    try:
        # Enhanced search parameters
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "num": str(min(max_results, 10)),  # Google max is 10
            "gl": "us",
            "hl": "en",
            "safe": "active",
            "filter": "1"  # Remove duplicate results
        }
        
        logger.info(f"Searching SerpAPI for: {query}")
        
        # Make API request with timeout
        response = requests.get(
            "https://serpapi.com/search", 
            params=params,
            timeout=10
        )
        
        if response.status_code != 200:
            logger.error(f"SerpAPI request failed: {response.status_code} - {response.text}")
            return results
            
        data = response.json()
        
        # Check for API errors
        if "error" in data:
            logger.error(f"SerpAPI error: {data['error']}")
            return results
            
        organic_results = data.get("organic_results", [])
        logger.info(f"SerpAPI returned {len(organic_results)} results")
        
        for result in organic_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            
            if not all([title, snippet, link]):
                continue
                
            domain = extract_domain(link)
            
            # Calculate credibility score with SerpAPI boost
            base_credibility = CREDIBLE_DOMAINS.get(domain, 0.6)  # Higher default for SerpAPI
            
            # Boost credibility for SerpAPI results
            boosted_credibility = min(1.0, base_credibility + 0.1)
            
            search_result = SearchResult(
                title=title,
                snippet=snippet,
                url=link,
                source_domain=domain,
                credibility_score=boosted_credibility
            )
            results.append(search_result)
            
        return results
        
    except requests.exceptions.Timeout:
        logger.error("SerpAPI request timed out")
        return results
    except requests.exceptions.RequestException as e:
        logger.error(f"SerpAPI request error: {str(e)}")
        return results
    except Exception as e:
        logger.error(f"Unexpected error in SerpAPI search: {str(e)}")
        return results

def search_with_google_custom(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Fallback Google Custom Search API when SerpAPI fails.
    """
    results = []
    
    try:
        params = {
            'key': GOOGLE_SEARCH_API_KEY,
            'cx': GOOGLE_SEARCH_ENGINE_ID,
            'q': query,
            'num': min(max_results, 10),
            'safe': 'active'
        }
        
        response = requests.get(
            'https://www.googleapis.com/customsearch/v1',
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            for item in items:
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                link = item.get('link', '')
                domain = extract_domain(link)
                
                search_result = SearchResult(
                    title=title,
                    snippet=snippet,
                    url=link,
                    source_domain=domain,
                    credibility_score=CREDIBLE_DOMAINS.get(domain, 0.5)
                )
                results.append(search_result)
                
    except Exception as e:
        logger.error(f"Google Custom Search error: {str(e)}")
    
    return results

def search_claim_evidence(claim: Claim, max_results: int = 8) -> List[SearchResult]:
    """
    Enhanced evidence search prioritizing SerpAPI with fallbacks.
    """
    try:
        all_results = []
        
        # 1. Primary: SerpAPI Search (Higher quota)
        serp_results = search_with_serpapi(claim.text, max_results=max_results)
        if serp_results:
            all_results.extend(serp_results)
            logger.info(f"SerpAPI found {len(serp_results)} results")
        else:
            logger.warning("SerpAPI returned no results, trying fallback")
            
            # 2. Fallback: Google Custom Search
            google_results = search_with_google_custom(claim.text, max_results=max_results//2)
            if google_results:
                all_results.extend(google_results)
                logger.info(f"Google Custom Search found {len(google_results)} results")
        
        # 3. Last resort: Wikipedia (only if we have very few results)
        if len(all_results) < 3:
            wiki_results = search_wikipedia_evidence(claim.text, max_results=3)
            all_results.extend(wiki_results)
            logger.info(f"Wikipedia found {len(wiki_results)} results")
        
        # Remove duplicates and sort by credibility
        unique_results = []
        seen_urls = set()
        
        for result in all_results:
            if result.url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result.url)
        
        # Sort by credibility score (descending)
        unique_results.sort(key=lambda x: x.credibility_score, reverse=True)
        
        return unique_results[:max_results]
        
    except Exception as e:
        logger.error(f"Error searching for claim evidence: {str(e)}")
        return []

def search_wikipedia_evidence(query: str, max_results: int = 3) -> List[SearchResult]:
    """Search Wikipedia for evidence about a claim."""
    results = []
    
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(query, results=max_results * 2)
        
        for title in search_results[:max_results]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary = page.summary[:500]
                
                result = SearchResult(
                    title=page.title,
                    snippet=summary,
                    url=page.url,
                    source_domain="wikipedia.org",
                    credibility_score=0.85
                )
                results.append(result)
                
            except (wikipedia.exceptions.DisambiguationError, 
                   wikipedia.exceptions.PageError, Exception) as e:
                logger.warning(f"Wikipedia error for {title}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {str(e)}")
    
    return results

def analyze_claim_with_evidence(claim: Claim, evidence: List[SearchResult]) -> FactCheckResult:
    """Analyze a claim against evidence using enhanced semantic analysis."""
    try:
        return enhanced_semantic_analysis(claim, evidence)
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        return basic_semantic_analysis(claim, evidence)

def enhanced_semantic_analysis(claim: Claim, evidence: List[SearchResult]) -> FactCheckResult:
    """
    Enhanced analysis that prioritizes SerpAPI results and non-Wikipedia sources.
    """
    llm = get_llm()
    if not llm:
        logger.warning("LLM not available, falling back to basic analysis")
        return basic_semantic_analysis(claim, evidence)

    try:
        # Calculate semantic similarities
        similarities = calculate_evidence_similarity(claim.text, evidence)
        
        # Enhanced scoring that heavily favors non-Wikipedia sources
        evidence_scores = []
        for i, result in enumerate(evidence):
            similarity = similarities[i] if i < len(similarities) else 0.0
            
            # Heavily prioritize non-Wikipedia sources
            if result.source_domain == "wikipedia.org":
                # Wikipedia gets lower combined score
                combined_score = (similarity * 0.3) + (result.credibility_score * 0.2)
            else:
                # Non-Wikipedia sources get much higher weighting
                combined_score = (similarity * 0.4) + (result.credibility_score * 0.8)
                
                # Extra boost for high-credibility domains
                if result.credibility_score > 0.85:
                    combined_score *= 1.2
            
            evidence_scores.append((result, combined_score))
        
        # Sort by combined score and take top evidence
        evidence_scores.sort(key=lambda x: x[1], reverse=True)
        top_evidence = [e[0] for e in evidence_scores[:5]]  # Top 5 pieces of evidence

        # Enhanced analysis prompt with source prioritization
        analysis_prompt = f"""
Analyze this claim against the provided evidence with extreme attention to detail.

CLAIM TO VERIFY: {claim.text}
CLAIM CATEGORY: {claim.category}

EVIDENCE SOURCES (ordered by reliability):
{chr(10).join(f'SOURCE {i+1} - {source.source_domain.upper()} (Credibility: {source.credibility_score:.1%}):\nTitle: {source.title}\nContent: {source.snippet}\nURL: {source.url}\n' for i, source in enumerate(top_evidence))}

ANALYSIS INSTRUCTIONS:
1. Prioritize evidence from non-Wikipedia sources (news, academic, government sites)
2. Give higher weight to sources with credibility scores above 85%
3. Look for direct factual matches, not just topical relevance
4. Consider multiple sources that corroborate the same information
5. Be conservative with TRUE verdicts - require strong evidence

VERDICT CRITERIA:
- "TRUE": Claim is directly supported by multiple high-credibility sources
- "FALSE": Claim is directly contradicted by credible sources  
- "UNCERTAIN": Insufficient evidence, conflicting information, or only Wikipedia sources

Provide your analysis in JSON format:
{{
    "VERDICT": "TRUE/FALSE/UNCERTAIN",
    "CONFIDENCE": <float between 0.0 and 1.0>,
    "EXPLANATION": "Detailed reasoning focusing on source quality and evidence strength",
    "EVIDENCE_SUMMARY": "Summary of key supporting/contradicting evidence from top sources"
}}
"""

        # Get LLM analysis
        response = llm.invoke(analysis_prompt)
        
        try:
            # Parse JSON response
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            analysis = json.loads(response_text)
            
            # Validate verdict
            verdict = analysis.get('VERDICT', 'UNCERTAIN').upper()
            if verdict not in ['TRUE', 'FALSE', 'UNCERTAIN']:
                verdict = 'UNCERTAIN'
            
            # Validate and adjust confidence
            confidence = float(analysis.get('CONFIDENCE', 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            # Boost confidence for non-Wikipedia sources
            non_wiki_sources = sum(1 for e in top_evidence if e.source_domain != "wikipedia.org")
            if non_wiki_sources >= 2:
                confidence = min(1.0, confidence * 1.3)
            elif non_wiki_sources == 0:
                # Penalize Wikipedia-only results
                confidence = max(0.0, confidence * 0.6)
            
            return FactCheckResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                explanation=analysis.get('EXPLANATION', 'No detailed explanation provided'),
                sources=top_evidence,
                evidence_summary=analysis.get('EVIDENCE_SUMMARY', 'No evidence summary provided')
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return basic_semantic_analysis(claim, evidence)
            
    except Exception as e:
        logger.error(f"Error in enhanced semantic analysis: {str(e)}")
        return basic_semantic_analysis(claim, evidence)

def fact_check_single_claim_enhanced(claim: Claim) -> FactCheckResult:
    """
    Enhanced single claim fact-checking with better error handling and retry logic.
    """
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            # Search for evidence with retry
            with st.spinner(f"🔍 Searching for evidence (attempt {attempt + 1})..."):
                evidence = search_claim_evidence(claim)
            
            if not evidence:
                if attempt < max_retries - 1:
                    st.warning(f"No evidence found, retrying...")
                    time.sleep(2)
                    continue
                else:
                    return FactCheckResult(
                        claim=claim,
                        verdict="UNCERTAIN",
                        confidence=0.0,
                        explanation="No evidence found through web search after multiple attempts.",
                        sources=[],
                        evidence_summary="No sources available for verification"
                    )
            
            # Analyze claim with evidence
            with st.spinner("🤖 Analyzing claim against evidence..."):
                result = analyze_claim_with_evidence(claim, evidence)
                
                # Log results for debugging
                logger.info(f"Fact-check result: {result.verdict} (confidence: {result.confidence:.2f})")
                logger.info(f"Sources used: {[s.source_domain for s in result.sources]}")
                
                return result
                
        except Exception as e:
            logger.error(f"Error in fact-checking attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                st.warning(f"Error occurred, retrying... ({str(e)})")
                time.sleep(2)
                continue
            else:
                return FactCheckResult(
                    claim=claim,
                    verdict="UNCERTAIN",
                    confidence=0.0,
                    explanation=f"Fact-checking failed after {max_retries} attempts: {str(e)}",
                    sources=[],
                    evidence_summary="Technical error prevented verification"
                )
    
    # This shouldn't be reached, but just in case
    return FactCheckResult(
        claim=claim,
        verdict="UNCERTAIN",
        confidence=0.0,
        explanation="Unknown error in fact-checking process",
        sources=[],
        evidence_summary="Fact-checking process failed"
    )

# Add these new functions
def calculate_evidence_similarity(claim_text: str, evidence: List[SearchResult]) -> List[float]:
    """Calculate semantic similarity between claim and evidence using TF-IDF."""
    if not evidence:
        return []
    
    try:
        texts = [claim_text] + [result.snippet for result in evidence]
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        claim_vector = tfidf_matrix[0]
        evidence_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(claim_vector, evidence_vectors)[0]
        
        return similarities.tolist()
        
    except Exception as e:
        logger.warning(f"Error calculating similarity: {str(e)}")
        return [0.0] * len(evidence)

def basic_semantic_analysis(claim: Claim, evidence: List[SearchResult]) -> FactCheckResult:
    """Fallback analysis when LLM is unavailable."""
    if not evidence:
        return FactCheckResult(
            claim=claim,
            verdict="UNCERTAIN",
            confidence=0.0,
            explanation="No evidence found for verification",
            sources=[],
            evidence_summary="No sources available"
        )
    
    claim_words = set(claim.text.lower().split())
    total_credibility = 0
    matching_evidence = 0
    
    for result in evidence:
        evidence_words = set(result.snippet.lower().split())
        word_overlap = len(claim_words.intersection(evidence_words))
        
        if word_overlap > 2:
            matching_evidence += 1
            total_credibility += result.credibility_score
    
    if matching_evidence > 0:
        avg_credibility = total_credibility / matching_evidence
        confidence = min(0.8, avg_credibility * (matching_evidence / len(evidence)))
        verdict = "TRUE" if confidence > 0.5 else "UNCERTAIN"
    else:
        verdict = "UNCERTAIN"
        confidence = 0.2
    
    return FactCheckResult(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        explanation=f"Based on keyword matching with {matching_evidence} relevant sources",
        sources=evidence,
        evidence_summary=f"Found {len(evidence)} sources, {matching_evidence} with relevant content"
    )

# UI COMPONENTS

def render_fact_check_results(results: List[FactCheckResult]):
    """Render fact-checking results in the UI."""
    if not results:
        st.info("No fact-check results available.")
        return
    
    # Summary statistics
    true_count = sum(1 for r in results if r.verdict == "TRUE")
    false_count = sum(1 for r in results if r.verdict == "FALSE")
    uncertain_count = sum(1 for r in results if r.verdict == "UNCERTAIN")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", len(results))
    with col2:
        st.metric("✅ True", true_count)
    with col3:
        st.metric("❌ False", false_count)
    with col4:
        st.metric("⚠️ Uncertain", uncertain_count)
    
    st.markdown("---")
    
    # Individual results
    for i, result in enumerate(results):
        # Color coding
        if result.verdict == "TRUE":
            border_color = "#28a745"  # Green
            emoji = "✅"
        elif result.verdict == "FALSE":
            border_color = "#dc3545"  # Red
            emoji = "❌"
        else:
            border_color = "#ffc107"  # Yellow
            emoji = "⚠️"
        
        # Create expandable card
        with st.expander(f"{emoji} **{result.verdict}** - {result.claim.text[:100]}{'...' if len(result.claim.text) > 100 else ''}"):
            
            # Claim details
            st.markdown(f"**Full Claim:** {result.claim.text}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Category:** {result.claim.category}")
            with col2:
                confidence_color = "green" if result.confidence > 0.7 else "orange" if result.confidence > 0.4 else "red"
                st.markdown(f"**Confidence:** :{confidence_color}[{result.confidence:.1%}]")
            
            # Explanation
            st.markdown("**Analysis:**")
            st.write(result.explanation)
            
            # Evidence summary
            if result.evidence_summary:
                st.markdown("**Evidence Summary:**")
                st.write(result.evidence_summary)
            
            # Sources
            if result.sources:
                st.markdown("**Sources:**")
                for j, source in enumerate(result.sources[:3]):  # Show top 3 sources
                    credibility_color = "green" if source.credibility_score > 0.8 else "orange" if source.credibility_score > 0.6 else "red"
                    st.markdown(f"{j+1}. [{source.title}]({source.url})")
                    st.markdown(f"   📍 {source.source_domain} (:{credibility_color}[Credibility: {source.credibility_score:.1%}])")
                    st.markdown(f"   💬 {source.snippet}")
                    st.markdown("")

# MAIN APPLICATION

# Main UI Layout
st.title("🎬 YouTube Video Q&A Bot with Fact Checker")
st.markdown("Extract insights from YouTube videos and automatically verify factual claims.")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📺 Video Processing", "💬 Q&A Chat", "🔍 Source Finder"])

with tab1:
    st.header("Process YouTube Video")
    
    # Input for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL:", key="url_input")
    
    # Process button
    if st.button("Process Video"):
        if youtube_url:
            try:
                # Reset session state
                st.session_state.chat_history = []
                st.session_state.extracted_claims = []
                st.session_state.fact_check_results = []
                
                video_id = extract_video_id(youtube_url)
                if not video_id:
                    st.error("Invalid YouTube URL. Please enter a valid YouTube URL.")
                else:
                    # Get video title
                    video_title = get_video_title(video_id)

                    # Get raw transcript
                    with st.spinner("Fetching transcript from YouTube..."):
                        raw_transcription = get_youtube_transcript(video_id)
                        st.session_state.raw_transcription = raw_transcription

                        if raw_transcription:
                            st.success("✅ Raw transcript fetched successfully!")

                            # Polish the transcript
                            with st.spinner("🔧 Enhancing transcript quality with AI..."):
                                polished_transcript = polish_transcript_with_gemini(raw_transcription, video_title)
                                if polished_transcript:
                                    st.success("✨ Transcript polished successfully!")
                                    
                                    # Store both versions in session state
                                    st.session_state.transcription = polished_transcript
                                    st.session_state.video_title = video_title
                                    st.session_state.video_url = youtube_url

                                    # Create document chunks from POLISHED transcript
                                    with st.spinner("📄 Creating document chunks..."):
                                        docs = split_text_into_documents(polished_transcript, video_title)
                                        if docs:
                                            st.success(f"Split transcript into {len(docs)} chunks")

                                            # Create and store embeddings from polished chunks
                                            vector_store, video_namespace = create_and_store_embeddings(docs, video_title)
                                            if vector_store and video_namespace:
                                                st.session_state.vector_store = vector_store
                                                st.session_state.video_namespace = video_namespace
                                                st.session_state.transcript_docs = docs
                                                
                                                # Set up QA chain with polished content
                                                qa_chain = setup_qa_chain(vector_store)
                                                if qa_chain:
                                                    st.session_state.conversation = qa_chain
                                                    st.success("🎉 Processing complete! Ready for Q&A and fact-checking.")
                                                else:
                                                    st.error("Failed to set up QA chain.")
                                            else:
                                                st.error("Failed to create vector store.")
                                        else:
                                            st.error("Failed to process transcript into documents.")
                                else:
                                    st.error("Failed to polish transcript.")
                        else:
                            st.error("Failed to fetch transcript. Please try another video with available captions.")

            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a YouTube URL.")

    # Display video info if available
    if st.session_state.transcription and st.session_state.video_title:
        st.markdown("---")
        st.subheader("📹 Video Information")
        st.markdown(f"**Title:** {st.session_state.video_title}")
        st.markdown(f"**URL:** [{st.session_state.video_url}]({st.session_state.video_url})")
        
        # Transcript preview
        with st.expander("📄 View Full Transcript"):
            st.text_area("Transcript:", st.session_state.transcription, height=400, disabled=True)

with tab2:
    st.header("💬 Chat with the Video")
    
    if not hasattr(st.session_state, 'conversation') or st.session_state.conversation is None:
        st.info("👆 Please process a video first in the 'Video Processing' tab.")
    else:
        st.markdown(f"**Currently discussing:** {st.session_state.video_title}")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.chat_message("user").write(message.content)
                else:
                    st.chat_message("assistant").write(message.content)

        # Get user question
        user_question = st.chat_input("Ask a question about the video...")

        if user_question:
            # Add user message to chat UI
            with chat_container:
                st.chat_message("user").write(user_question)

            # Add to session state history
            st.session_state.chat_history.append(HumanMessage(content=user_question))

            # Get the answer
            answer, sources = process_query(user_question, st.session_state.conversation)

            # Display assistant response
            with chat_container:
                assistant_message = st.chat_message("assistant")
                assistant_message.write(answer)

                # Display sources if available
                if sources:
                    with assistant_message.expander("📚 Sources from transcript"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i + 1}:**")
                            st.markdown(doc.page_content)
                            if i < len(sources) - 1:
                                st.markdown("---")

            # Add to session state history
            st.session_state.chat_history.append(AIMessage(content=answer))

with tab3:
    st.header("🔍 Find Sources for Claims")
    
    if not hasattr(st.session_state, 'transcript_docs') or st.session_state.transcript_docs is None:
        st.info("👆 Please process a video first in the 'Video Processing' tab.")
    else:
        st.markdown(f"**Analyzing claims from:** {st.session_state.video_title}")
        
        # Extract claims section
        if st.button("📝 Extract Claims from Transcript"):
            try:
                with st.spinner("🔍 Analyzing transcript for factual claims..."):
                    claims = extract_claims_from_transcript(st.session_state.transcript_docs)
                    st.session_state.extracted_claims = claims
                
                if claims:
                    st.success(f"✅ Found {len(claims)} claims to check!")
                else:
                    st.warning("⚠️ No specific claims found in this transcript.")
                    
            except Exception as e:
                st.error(f"Error extracting claims: {str(e)}")
        
        st.markdown("---")
        
        # Display extracted claims with source finder
        if hasattr(st.session_state, 'extracted_claims') and st.session_state.extracted_claims:
            render_individual_claims_ui(st.session_state.extracted_claims)
        else:
            st.info("📝 Click 'Extract Claims from Transcript' to find statements to check.")

# Sidebar - Configuration and Help
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Debug mode toggle
    debug_mode = st.toggle("🐛 Debug Mode", value=False)
    
    # API Status
    st.subheader("📡 API Status")
    try:
        # Test Google API
        test_llm = get_llm()
        if test_llm:
            st.success("✅ Google Gemini API")
        else:
            st.error("❌ Google Gemini API")
    except:
        st.error("❌ Google Gemini API")
    
    try:
        # Test Search API
        test_response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={'key': GOOGLE_SEARCH_API_KEY, 'cx': GOOGLE_SEARCH_ENGINE_ID, 'q': 'test'},
            timeout=5
        )
        if test_response.status_code in [200, 400]:  # 400 is expected for empty query
            st.success("✅ Google Search API")
        else:
            st.error("❌ Google Search API")
    except:
        st.error("❌ Google Search API")
    
    try:
        # Test Pinecone
        if int(pinecone.__version__.split('.')[0]) >= 4:
            pc.list_indexes()
        else:
            pc.list_indexes()
        st.success("✅ Pinecone API")
    except:
        st.error("❌ Pinecone API")
    
    st.markdown("---")
    
    # Help Section
    with st.expander("❓ How to Setup"):
        st.markdown("""
        ### Required API Keys in secrets.toml:
        
        ```toml
        # Google APIs
        GOOGLE_API_KEY = "your-gemini-api-key"
        GOOGLE_SEARCH_API_KEY = "your-custom-search-api-key"
        GOOGLE_SEARCH_ENGINE_ID = "your-search-engine-id"
        
        # Pinecone
        PINECONE_API_KEY = "your-pinecone-api-key"
        PINECONE_ENVIRONMENT = "gcp-starter"
        ```
        
        ### Get API Keys:
        - **Google Gemini**: [AI Studio](https://makersuite.google.com/app/apikey)
        - **Google Search**: [Google Cloud Console](https://console.cloud.google.com/)
        - **Search Engine**: [Programmable Search](https://programmablesearchengine.google.com/)
        - **Pinecone**: [Pinecone Console](https://app.pinecone.io)
        """)
    
    with st.expander("🎯 How It Works"):
        st.markdown("""
        ### Fact-Checking Process:
        
        1. **Claim Extraction**: AI analyzes transcript chunks to identify factual statements
        2. **Evidence Gathering**: Google Search API finds relevant sources
        3. **Source Evaluation**: Sources ranked by credibility (academic, news, gov sites)
        4. **AI Analysis**: Gemini compares claims against evidence
        5. **Verdict**: Claims marked as TRUE, FALSE, or UNCERTAIN
        
        ### Credibility Scoring:
        - 🏛️ Government sites: 95%
        - 🎓 Educational (.edu): 90%
        - 📰 Major news outlets: 70-85%
        - 📚 Wikipedia: 80%
        - 🔬 Scientific journals: 85-90%
        """)

# Debug Panel
if debug_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🐛 Debug Info")
    
    if hasattr(st.session_state, 'vector_store') and st.session_state.vector_store:
        st.sidebar.write(f"📊 Index: {PINECONE_INDEX_NAME}")
        st.sidebar.write(f"🏷️ Namespace: {st.session_state.video_namespace}")
    
    st.sidebar.write(f"💬 Chat messages: {len(st.session_state.chat_history)}")
    st.sidebar.write(f"🔍 Claims analyzed: {len(st.session_state.fact_check_results)}")
    
    # Show recent logs
    try:
        with open("youtube_qa_bot.log", "r") as log_file:
            logs = log_file.readlines()
            if logs:
                st.sidebar.text_area("📝 Recent Logs", "".join(logs[-5:]), height=100)
    except:
        st.sidebar.info("No logs available")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.7;'>
    <p>🚀 Enhanced YouTube Q&A Bot with AI-Powered Fact Checking</p>
    <p>Built with Streamlit • LangChain • Pinecone • Google Gemini • Custom Search API</p>
</div>
""", unsafe_allow_html=True)
