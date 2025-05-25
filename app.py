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
    'wikipedia.org': 0.8, 'edu': 0.9, 'gov': 0.95, 'nature.com': 0.9,
    'sciencedirect.com': 0.85, 'pubmed.ncbi.nlm.nih.gov': 0.9,
    'bbc.com': 0.8, 'reuters.com': 0.85, 'ap.org': 0.85,
    'cnn.com': 0.7, 'nytimes.com': 0.8, 'washingtonpost.com': 0.8
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
    """Extract factual claims from transcript chunks using Gemini."""
    llm = get_llm()
    if not llm:
        return []
    
    claims = []
    
    claim_extraction_prompt = """
    Analyze the following transcript chunk and extract all factual claims or statements that can be verified. 
    Focus on:
    - Specific statistics, numbers, dates, or measurements
    - Historical facts or events
    - Scientific claims or assertions
    - Names of people, places, organizations
    - Claims about cause and effect relationships
    - Specific product claims or comparisons
    
    Ignore:
    - Personal opinions or subjective statements
    - General advice or recommendations
    - Hypothetical scenarios
    - Questions
    
    For each claim found, respond with a JSON object containing:
    - "claim": the exact factual statement
    - "category": type of claim (historical, scientific, statistical, etc.)
    
    If no verifiable claims are found, respond with an empty JSON array.
    
    Transcript chunk:
    {chunk_content}
    
    Response format: JSON array of claim objects
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
    """Search for evidence about a claim using Google Custom Search API."""
    try:
        # Prepare search query
        search_query = claim.text[:100]  # Limit query length
        encoded_query = quote_plus(search_query)
        
        # Google Custom Search API endpoint
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_SEARCH_API_KEY,
            'cx': GOOGLE_SEARCH_ENGINE_ID,
            'q': encoded_query,
            'num': max_results
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        search_data = response.json()
        results = []
        
        if 'items' in search_data:
            for item in search_data['items']:
                # Calculate credibility score
                domain = item.get('displayLink', '')
                credibility_score = 0.5  # Default score
                
                for credible_domain, score in CREDIBLE_DOMAINS.items():
                    if credible_domain in domain.lower():
                        credibility_score = score
                        break
                
                result = SearchResult(
                    title=item.get('title', ''),
                    snippet=item.get('snippet', ''),
                    url=item.get('link', ''),
                    source_domain=domain,
                    credibility_score=credibility_score
                )
                results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching for claim evidence: {str(e)}")
        return []

def analyze_claim_with_evidence(claim: Claim, evidence: List[SearchResult]) -> FactCheckResult:
    """Analyze a claim against evidence using Gemini to determine truth value."""
    llm = get_llm()
    if not llm:
        return FactCheckResult(
            claim=claim,
            verdict="UNCERTAIN",
            confidence=0.0,
            explanation="Could not analyze claim due to LLM unavailability",
            sources=evidence,
            evidence_summary="No analysis available"
        )
    
    analysis_prompt = """
    You are a fact-checking expert. Analyze the following claim against the provided evidence and determine if it's TRUE, FALSE, or UNCERTAIN.

    CLAIM TO VERIFY:
    "{claim_text}"

    EVIDENCE FROM WEB SEARCH:
    {evidence_text}

    Instructions:
    1. Compare the claim against the evidence provided
    2. Consider the credibility of sources (higher credibility = more weight)
    3. Look for consensus across multiple sources
    4. Be conservative - if evidence is mixed or insufficient, mark as UNCERTAIN

    Respond with a JSON object containing:
    - "verdict": "TRUE", "FALSE", or "UNCERTAIN"
    - "confidence": float between 0.0 and 1.0
    - "explanation": detailed explanation of your reasoning
    - "evidence_summary": brief summary of key evidence points

    JSON Response:
    """
    
    # Format evidence for prompt
    evidence_text = ""
    for i, result in enumerate(evidence, 1):
        evidence_text += f"{i}. Source: {result.source_domain} (Credibility: {result.credibility_score})\n"
        evidence_text += f"   Title: {result.title}\n"
        evidence_text += f"   Content: {result.snippet}\n\n"
    
    try:
        prompt = analysis_prompt.format(
            claim_text=claim.text,
            evidence_text=evidence_text
        )
        
        response = llm.invoke(prompt)
        
        # Parse JSON response
        try:
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            analysis_data = json.loads(response_text)
            
            return FactCheckResult(
                claim=claim,
                verdict=analysis_data.get('verdict', 'UNCERTAIN'),
                confidence=float(analysis_data.get('confidence', 0.0)),
                explanation=analysis_data.get('explanation', 'No explanation provided'),
                sources=evidence,
                evidence_summary=analysis_data.get('evidence_summary', 'No summary available')
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON from analysis response: {e}")
            # Fallback analysis
            response_text = response.content.lower()
            if 'true' in response_text and 'false' not in response_text:
                verdict = "TRUE"
                confidence = 0.6
            elif 'false' in response_text and 'true' not in response_text:
                verdict = "FALSE"
                confidence = 0.6
            else:
                verdict = "UNCERTAIN"
                confidence = 0.3
            
            return FactCheckResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                explanation=response.content,
                sources=evidence,
                evidence_summary="Analysis based on text parsing"
            )
            
    except Exception as e:
        logger.error(f"Error analyzing claim: {str(e)}")
        return FactCheckResult(
            claim=claim,
            verdict="UNCERTAIN",
            confidence=0.0,
            explanation=f"Error during analysis: {str(e)}",
            sources=evidence,
            evidence_summary="Analysis failed"
        )

def perform_fact_checking(transcript_chunks: List[Document]) -> List[FactCheckResult]:
    """Main fact-checking pipeline."""
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
tab1, tab2, tab3 = st.tabs(["📺 Video Processing", "💬 Q&A Chat", "🔍 Fact Check"])

with tab1:
    st.header("Process YouTube Video")
    
    # Input for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL:", key="url_input")
    
    # Process button
    if st.button("Process Video"):
        if youtube_url:
            try:
                # Reset session state for new video
                st.session_state.chat_history = []
                st.session_state.extracted_claims = []
                st.session_state.fact_check_results = []
                
                # Extract video ID
                video_id = extract_video_id(youtube_url)
                if not video_id:
                    st.error("Invalid YouTube URL. Please enter a valid YouTube URL.")
                else:
                    # Get video title
                    video_title = get_video_title(video_id)

                    # Get transcript directly from YouTube
                    with st.spinner("Fetching transcript from YouTube..."):
                        transcription = get_youtube_transcript(video_id)

                        if transcription:
                            st.success("Transcript fetched successfully!")

                            # Store in session state
                            st.session_state.transcription = transcription
                            st.session_state.video_title = video_title
                            st.session_state.video_url = youtube_url

                            # Split text into documents
                            docs = split_text_into_documents(transcription, video_title)

                            if docs:
                                st.success(f"Split transcript into {len(docs)} chunks")

                                # Create and store embeddings
                                vector_store, video_namespace = create_and_store_embeddings(docs, video_title)
                                if vector_store and video_namespace:
                                    st.session_state.vector_store = vector_store
                                    st.session_state.video_namespace = video_namespace
                                    st.success(f"Created embeddings and stored in Pinecone index under namespace: {video_namespace}")

                                    # Set up QA chain
                                    qa_chain = setup_qa_chain(vector_store)
                                    if qa_chain:
                                        st.session_state.conversation = qa_chain
                                        st.success("✅ Ready to answer questions about this video!")
                                        
                                        # Store docs for fact-checking
                                        st.session_state.transcript_docs = docs
                                        
                                        st.info("💡 You can now switch to the Q&A tab to ask questions or the Fact Check tab to verify claims!")
                                    else:
                                        st.error("Failed to set up QA chain. Please try again.")
                                else:
                                    st.error("Failed to create vector store. Please try again.")
                            else:
                                st.error("Failed to process the transcript into documents.")
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
        
        # Fact-checking controls
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔍 Start Fact Checking", disabled=st.session_state.fact_check_in_progress):
                st.session_state.fact_check_in_progress = True
                try:
                    # Perform fact-checking
                    results = perform_fact_checking(st.session_state.transcript_docs)
                    st.session_state.fact_check_results = results
                    st.success(f"✅ Fact-checking complete! Analyzed {len(results)} claims.")
                except Exception as e:
                    st.error(f"Error during fact-checking: {str(e)}")
                finally:
                    st.session_state.fact_check_in_progress = False
        
        with col2:
            if st.session_state.fact_check_results:
                if st.button("📊 Export Results"):
                    # Create export data
                    export_data = []
                    for result in st.session_state.fact_check_results:
                        export_data.append({
                            "claim": result.claim.text,
                            "verdict": result.verdict,
                            "confidence": f"{result.confidence:.1%}",
                            "explanation": result.explanation,
                            "evidence_summary": result.evidence_summary,
                            "sources": [{"title": s.title, "url": s.url, "domain": s.source_domain} for s in result.sources[:3]]
                        })
                    
                    # Convert to JSON for download
                    import json
                    json_data = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json_data,
                        file_name=f"fact_check_report_{st.session_state.video_title[:30]}.json",
                        mime="application/json"
                    )
        
        # Display progress if fact-checking is in progress
        if st.session_state.fact_check_in_progress:
            st.info("🔄 Fact-checking in progress... This may take a few minutes.")
        
        # Display results
        if st.session_state.fact_check_results:
            st.markdown("---")
            render_fact_check_results(st.session_state.fact_check_results)
        elif not st.session_state.fact_check_in_progress:
            st.info("Click 'Start Fact Checking' to analyze claims from the video transcript.")

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
