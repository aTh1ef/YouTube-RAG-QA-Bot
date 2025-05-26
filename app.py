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
warnings.filterwarnings("ignore")

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
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
import random
from requests.exceptions import RequestException

# Data classes for fact-checking
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
    credibility_score: float = 0.0

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
CREDIBLE_DOMAINS = {
    'wikipedia.org': 0.85, 'edu': 0.9, 'gov': 0.95, 
    'nature.com': 0.9, 'sciencedirect.com': 0.85, 
    'pubmed.ncbi.nlm.nih.gov': 0.9, 'bbc.com': 0.8, 
    'reuters.com': 0.85, 'ap.org': 0.85, 'cnn.com': 0.7, 
    'nytimes.com': 0.8, 'washingtonpost.com': 0.8,
    'duckduckgo.com': 0.6, 'britannica.com': 0.8,
    'nationalgeographic.com': 0.75, 'smithsonianmag.com': 0.75
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

# Add these at the top with other constants
PROXY_LIST = [
    None,  # Try without proxy first
    'http://proxy1.example.com:8080',  # Replace with actual proxies
    'http://proxy2.example.com:8080',
    'http://proxy3.example.com:8080'
]

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def get_youtube_transcript(video_id):
    """
    Get transcript with proxy support and retry mechanism.
    """
    formatter = TextFormatter()
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            # Randomly select a proxy (None means no proxy)
            current_proxy = random.choice(PROXY_LIST)
            
            if current_proxy:
                st.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Using proxy to fetch transcript...")
                proxies = {
                    'http': current_proxy,
                    'https': current_proxy
                }
                transcript_list = YouTubeTranscriptApi.get_transcript(
                    video_id,
                    proxies=proxies
                )
            else:
                st.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Fetching transcript directly...")
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            if not transcript_list:
                st.warning("No transcript content available for this video.")
                return None

            # Format transcript
            formatted_transcript = formatter.format_transcript(transcript_list)
            if not formatted_transcript.strip():
                st.warning("Transcript is empty after processing.")
                return None

            return formatted_transcript

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            st.error("Transcripts are disabled or not available for this video.")
            return None
            
        except Exception as e:
            last_error = str(e)
            if "too many requests" in last_error.lower() or "blocked" in last_error.lower():
                # If this is not the last attempt, wait and try again
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (attempt + 1)  # Exponential backoff
                    st.warning(f"YouTube API rate limit hit. Waiting {delay} seconds before retrying...")
                    time.sleep(delay)
                    continue
            else:
                # For other errors, log and continue retrying
                logger.warning(f"Error fetching transcript (attempt {attempt + 1}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue

    # If we get here, all retries failed
    st.error(f"Failed to fetch transcript after {MAX_RETRIES} attempts. Last error: {last_error}")
    logger.error(f"All transcript fetch attempts failed. Last error: {last_error}")
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

def search_claim_evidence(claim: Claim, max_results: int = 5) -> List[SearchResult]:
    """Search for evidence about a claim using free sources."""
    try:
        results = []
        
        # 1. Wikipedia Search (Primary source)
        wiki_results = search_wikipedia_evidence(claim.text, max_results=3)
        results.extend(wiki_results)
        
        # 2. DuckDuckGo Search (Secondary source)
        ddg_results = search_duckduckgo_evidence(claim.text, max_results=2)
        results.extend(ddg_results)
        
        # Remove duplicates and sort by credibility
        unique_results = []
        seen_urls = set()
        
        for result in results:
            if result.url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result.url)
        
        # Sort by credibility score
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

def search_duckduckgo_evidence(query: str, max_results: int = 2) -> List[SearchResult]:
    """Search DuckDuckGo for evidence."""
    results = []
    
    try:
        ddg_url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(ddg_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('AbstractText'):
                result = SearchResult(
                    title=data.get('Heading', 'DuckDuckGo Result'),
                    snippet=data.get('AbstractText', ''),
                    url=data.get('AbstractURL', ''),
                    source_domain=data.get('AbstractSource', 'duckduckgo.com'),
                    credibility_score=0.6
                )
                results.append(result)
            
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and topic.get('Text'):
                    result = SearchResult(
                        title=topic.get('FirstURL', {}).get('Text', 'Related Topic'),
                        snippet=topic.get('Text', ''),
                        url=topic.get('FirstURL', {}).get('URL', ''),
                        source_domain='duckduckgo.com',
                        credibility_score=0.5
                    )
                    results.append(result)
                    
    except Exception as e:
        logger.warning(f"Error searching DuckDuckGo: {str(e)}")
    
    return results

def analyze_claim_with_evidence(claim: Claim, evidence: List[SearchResult]) -> FactCheckResult:
    """Analyze a claim against evidence using enhanced semantic analysis."""
    try:
        return enhanced_semantic_analysis(claim, evidence)
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        return basic_semantic_analysis(claim, evidence)

def perform_fact_checking(transcript_chunks: List[Document]) -> List[FactCheckResult]:
    # Main fact-checking pipeline.
    results = []
    
    # Step 1: Extract claims
    with st.spinner("Extracting claims from transcript..."):
        claims = extract_claims_from_transcript(transcript_chunks)
        st.success(f"Extracted {len(claims)} potential claims")
    
    if not claims:
        st.warning("No verifiable claims found in the transcript.")
        return results
    
    # Step 2: Process each claim
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, claim in enumerate(claims):
        try:
            status_text.text(f"Fact-checking claim {i+1} of {len(claims)}: {claim.text[:50]}...")
            
            # Search for evidence
            evidence = search_claim_evidence(claim)
            
            if evidence:
                # Analyze claim with evidence
                result = analyze_claim_with_evidence(claim, evidence)
                results.append(result)
            else:
                # No evidence found
                result = FactCheckResult(
                    claim=claim,
                    verdict="UNCERTAIN",
                    confidence=0.0,
                    explanation="No evidence found through web search",
                    sources=[],
                    evidence_summary="No sources available"
                )
                results.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(claims))
            
            # Rate limiting
            time.sleep(0.5)  # Avoid hitting API rate limits
            
        except Exception as e:
            logger.error(f"Error processing claim {i}: {str(e)}")
            continue
    
    status_text.text("Fact-checking complete!")
    progress_bar.empty()
    
    return results

def fact_check_single_claim(claim: Claim) -> FactCheckResult:
    """Fact-check a single claim on-demand."""
    try:
        # Search for evidence
        with st.spinner(f"🔍 Searching for evidence about: {claim.text[:50]}..."):
            evidence = search_claim_evidence(claim)
        
        if evidence:
            # Analyze claim with evidence
            with st.spinner("🤖 Analyzing claim against evidence..."):
                result = analyze_claim_with_evidence(claim, evidence)
            return result
        else:
            return FactCheckResult(
                claim=claim,
                verdict="UNCERTAIN",
                confidence=0.0,
                explanation="No evidence found through web search to verify this claim.",
                sources=[],
                evidence_summary="No sources available for verification"
            )
            
    except Exception as e:
        logger.error(f"Error fact-checking single claim: {str(e)}")
        return FactCheckResult(
            claim=claim,
            verdict="UNCERTAIN",
            confidence=0.0,
            explanation=f"Error during fact-checking: {str(e)}",
            sources=[],
            evidence_summary="Fact-checking failed due to technical error"
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

def enhanced_semantic_analysis(claim: Claim, evidence: List[SearchResult]) -> FactCheckResult:
    """
    Enhanced semantic analysis of claims using Gemini for more accurate fact-checking
    while preserving context and meaning.
    """
    llm = get_llm()
    if not llm:
        logger.warning("LLM not available, falling back to basic analysis")
        return basic_semantic_analysis(claim, evidence)

    try:
        # Calculate semantic similarities for evidence ranking
        similarities = calculate_evidence_similarity(claim.text, evidence)
        
        # Sort evidence by both similarity and credibility
        evidence_scores = []
        for i, result in enumerate(evidence):
            combined_score = (similarities[i] * 0.6) + (result.credibility_score * 0.4)
            evidence_scores.append((result, combined_score))
        
        evidence_scores.sort(key=lambda x: x[1], reverse=True)
        top_evidence = [e[0] for e in evidence_scores[:3]]  # Use top 3 most relevant pieces of evidence

        # Construct analysis prompt
        analysis_prompt = f"""
Analyze this claim against the provided evidence with extreme attention to detail and context preservation.

CLAIM TO VERIFY: {claim.text}
CLAIM CATEGORY: {claim.category}

EVIDENCE SOURCES:
{chr(10).join(f'SOURCE {i+1} ({source.source_domain}, Credibility: {source.credibility_score:.1%}):\n{source.snippet}\n' for i, source in enumerate(top_evidence))}

ANALYSIS INSTRUCTIONS:
1. Evaluate if the claim is fully supported, partially supported, contradicted, or unverifiable based on evidence
2. Consider source credibility and relevance
3. Look for specific numbers, dates, facts that can be directly compared
4. Note any context or qualifications that affect the claim's accuracy
5. Consider temporal context (when the claim was made vs. evidence)

OUTPUT FORMAT:
1. VERDICT: Exactly one of ["TRUE", "FALSE", "UNCERTAIN"]
2. CONFIDENCE: Number between 0.0 and 1.0
3. EXPLANATION: Detailed reasoning
4. EVIDENCE SUMMARY: Key supporting/contradicting points

Respond in JSON format only.
"""

        # Get LLM analysis
        response = llm.invoke(analysis_prompt)
        
        try:
            # Clean and parse JSON response
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            analysis = json.loads(response_text)
            
            # Validate and normalize verdict
            verdict = analysis.get('VERDICT', 'UNCERTAIN').upper()
            if verdict not in ['TRUE', 'FALSE', 'UNCERTAIN']:
                verdict = 'UNCERTAIN'
            
            # Validate confidence
            confidence = float(analysis.get('CONFIDENCE', 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            
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

def render_individual_claims_ui(claims: List[Claim]):
    """Render UI for individual claim fact-checking."""
    if not claims:
        st.info("No claims available for fact-checking.")
        return
    
    # Display each claim with a fact-check button
    for i, claim in enumerate(claims):
        with st.expander(f"Claim {i+1}: {claim.text[:100]}{'...' if len(claim.text) > 100 else ''}", expanded=False):
            # Display claim details
            st.markdown(f"**Full Claim:** {claim.text}")
            st.markdown(f"**Source Chunk:** {claim.source_chunk}")
            
            # Fact-check button
            if st.button(f"🔍 Fact-Check Claim {i+1}", key=f"factcheck_{i}"):
                with st.spinner("Fact-checking claim..."):
                    result = fact_check_single_claim(claim)
                
                # Display result
                if result:
                    render_single_claim_result(result)
                else:
                    st.error("Error fact-checking claim.")

def render_single_claim_result(result: FactCheckResult):
    """Render result for a single fact-checked claim."""
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
    
    # Result card
    with st.container():
        st.markdown(f"<div style='border: 2px solid {border_color}; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
        st.markdown(f"**Verdict: {result.verdict}** {emoji}")
        st.markdown(f"**Confidence:** {result.confidence:.1%}")
        st.markdown(f"**Explanation:** {result.explanation}")
        
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
        
        st.markdown("</div>", unsafe_allow_html=True)

# MAIN APPLICATION

# Main UI Layout
st.title("🎬 YouTube Video Q&A Bot with Fact Checker")
st.markdown("Extract insights from YouTube videos and automatically verify factual claims.")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📺 Video Processing", "💬 Q&A Chat", "🔍 Fact Check"])

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
    st.header("🔍 Fact Check Analysis")
    
    if not hasattr(st.session_state, 'transcript_docs') or st.session_state.transcript_docs is None:
        st.info("👆 Please process a video first in the 'Video Processing' tab.")
    else:
        st.markdown(f"**Analyzing claims from:** {st.session_state.video_title}")
        
        # Extract claims section
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("📝 Extract Claims from Transcript"):
                try:
                    with st.spinner("🔍 Analyzing transcript for factual claims..."):
                        claims = extract_claims_from_transcript(st.session_state.transcript_docs)
                        st.session_state.extracted_claims = claims
                        # Clear previous results when extracting new claims
                        st.session_state.claim_results = {}
                    
                    if claims:
                        st.success(f"✅ Extracted {len(claims)} verifiable claims!")
                    else:
                        st.warning("⚠️ No specific factual claims found in this transcript.")
                        
                except Exception as e:
                    st.error(f"Error extracting claims: {str(e)}")
        
        with col2:
            # Bulk fact-check option
            if st.session_state.extracted_claims:
                if st.button("🚀 Fact Check All", help="Fact-check all claims at once (uses more API calls)"):
                    try:
                        with st.spinner("⚡ Fact-checking all claims... This may take a while."):
                            progress_bar = st.progress(0)
                            
                            for i, claim in enumerate(st.session_state.extracted_claims):
                                claim_key = f"claim_{i}"
                                result = fact_check_single_claim(claim)
                                st.session_state.claim_results[claim_key] = result
                                progress_bar.progress((i + 1) / len(st.session_state.extracted_claims))
                                time.sleep(0.5)  # Rate limiting
                            
                            progress_bar.empty()
                        st.success("✅ All claims fact-checked!")
                        
                    except Exception as e:
                        st.error(f"Error during bulk fact-checking: {str(e)}")
        
        st.markdown("---")
        
        # Display extracted claims with individual fact-check buttons
        if st.session_state.extracted_claims:
            render_individual_claims_ui(st.session_state.extracted_claims)
        else:
            st.info("📝 Click 'Extract Claims from Transcript' to find factual statements that can be verified.")

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
