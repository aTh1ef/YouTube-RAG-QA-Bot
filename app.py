# app.py
import streamlit as st
import os
import requests
import re
import logging
from typing import List, Dict, Any

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
from pinecone import Pinecone  
from langchain_pinecone import PineconeVectorStore

# YouTube Transcript API for getting transcripts directly
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

# Set page configuration
st.set_page_config(
    page_title="YouTube Video Q&A Bot",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'video_title' not in st.session_state:
    st.session_state.video_title = None
if 'video_url' not in st.session_state:
    st.session_state.video_url = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'video_namespace' not in st.session_state:
    st.session_state.video_namespace = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load API keys from Streamlit secrets
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT", "gcp-starter")  # Provide a default
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

    # Initialize Pinecone client with the API key
    pinecone.init(api_key=PINECONE_API_KEY)
except Exception as e:
    st.error(f"Error loading API keys: {str(e)}")
    st.error("Please make sure you have set up your .streamlit/secrets.toml file with the required API keys.")
    st.stop()

# Pinecone specific configuration
PINECONE_INDEX_NAME = "youtube-qa-bot"


# Initialize HuggingFace Embeddings - cached to avoid reinitialization issues
@st.cache_resource
def get_embeddings_model():
    try:
        # Set environment variable to avoid torch issues with Streamlit
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Using a simpler embedding model with fewer issues
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=None
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {str(e)}")
        st.error(f"Failed to initialize embeddings model: {str(e)}")
        return None


# Initialize LLM - moved out of function to avoid reinitialization
@st.cache_resource
def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Updated to current model name
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2048
        )
    except Exception as e:
        logger.error(f"Error initializing primary LLM: {str(e)}")
        try:
            # Fallback to older model
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


# Function to extract YouTube video ID from URL
def extract_video_id(url):
    # Regular expression patterns for YouTube URLs
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/e\/|youtube\.com\/user\/.+\/|youtube\.com\/user\/(?!.+\/)|youtube\.com\/.*[?&]v=|youtube\.com\/.*[?&]vi=)([^#&?\/\s]{11})',
        r'(?:youtube\.com\/shorts\/|youtube\.com\/live\/)([^#&?\/\s]{11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


# Function to get video title from YouTube
def get_video_title(video_id):
    try:
        # Use a simple HTTP request to get the title from HTML (no API key needed)
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}")
        if response.status_code == 200:
            # Extract title from HTML meta tags
            match = re.search(r'<title>(.+?) - YouTube</title>', response.text)
            if match:
                return match.group(1)
        return f"YouTube Video {video_id}"
    except Exception as e:
        logger.warning(f"Could not get video title: {str(e)}")
        return f"YouTube Video {video_id}"


# Function to get transcript directly from YouTube
def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Check if the transcript is empty
        if not transcript_list:
            st.error("No transcript content available for this video.")
            return None

        # Combine all transcript parts into a single text
        full_transcript = " ".join([part["text"] + " " for part in transcript_list])

        # Check if the transcript is empty after cleaning
        if not full_transcript.strip():
            st.error("Transcript is empty after processing.")
            return None

        return full_transcript
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        # Generic exception handling for any other transcript-related errors
        st.error(f"No transcript available for this video: {str(e)}")
        return None


# Function to split text into documents using LangChain
def split_text_into_documents(text, video_title):
    try:
        # Create a LangChain Document
        doc = Document(page_content=text, metadata={"source": video_title})

        # Use LangChain's RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for better retrieval
            chunk_overlap=100,
            length_function=len,
        )

        # Split into chunks
        docs = text_splitter.split_documents([doc])

        # Add metadata to each chunk
        for i, chunk in enumerate(docs):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["video_title"] = video_title

        return docs
    except Exception as e:
        logger.error(f"Error splitting text into documents: {str(e)}")
        st.error(f"Error splitting text into documents: {str(e)}")
        return []


# Function to create embeddings and store in Pinecone using LangChain
def create_and_store_embeddings(docs, video_title):
    try:
        # Generate a unique namespace for this video
        video_namespace = ''.join(e for e in video_title if e.isalnum()).lower()[:40]
        video_namespace = video_namespace.replace(" ", "-")

        # Initialize the embedding model
        embeddings = get_embeddings_model()
        if embeddings is None:
            st.error("Failed to initialize embeddings model. Cannot continue.")
            return None, None

        # Check if Pinecone index exists, create if not
        try:
            index_names = pinecone.list_indexes()
            if PINECONE_INDEX_NAME not in index_names:
                st.warning(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new index...")
                try:
                    # Create a new index with the updated SDK
                    pinecone.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=384,  # Dimension for all-MiniLM-L6-v2
                        metric="cosine"
                    )
                    st.success(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
                except Exception as e:
                    logger.error(f"Failed to create Pinecone index: {str(e)}")
                    st.error(f"Failed to create Pinecone index: {str(e)}")
                    return None, None
        except Exception as e:
            logger.error(f"Error checking Pinecone indexes: {str(e)}")
            st.error(f"Error checking Pinecone indexes: {str(e)}")
            return None, None

        # Initialize Pinecone Vector Store with LangChain
        with st.spinner("Creating and storing embeddings..."):
            # Use LangChain's Pinecone integration with updated approach
            try:
                # Get the index directly using updated API
                index = pinecone.Index(PINECONE_INDEX_NAME)

                # Delete existing vectors in this namespace to avoid conflicts
                try:
                    index.delete(namespace=video_namespace, delete_all=True)
                    logger.info(f"Deleted existing vectors in namespace: {video_namespace}")
                except Exception as e:
                    logger.warning(f"No existing vectors to delete in namespace {video_namespace}: {str(e)}")

                # Use updated from_documents signature
                vector_store = PineconeVectorStore.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX_NAME,
                    namespace=video_namespace
                )
                logger.info(f"Successfully stored {len(docs)} document chunks in Pinecone")
            except Exception as e:
                logger.error(f"First attempt to store embeddings failed: {str(e)}")
                try:
                    # Alternative approach passing index directly
                    vector_store = PineconeVectorStore.from_documents(
                        documents=docs,
                        embedding=embeddings,
                        index=index,  # Use the index directly
                        namespace=video_namespace
                    )
                    logger.info(f"Successfully stored {len(docs)} document chunks in Pinecone (alternative method)")
                except Exception as e2:
                    logger.error(f"Both attempts to store embeddings failed: {str(e2)}")
                    st.error(f"Error storing embeddings: {str(e2)}")
                    return None, None

        return vector_store, video_namespace
    except Exception as e:
        logger.error(f"Error in create_and_store_embeddings: {str(e)}")
        st.error(f"Error storing embeddings: {str(e)}")
        return None, None


# Create a conversation memory
def setup_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Ensure this matches with chain's output_key
    )


# Create a retrieval-based conversation chain using LangChain
def setup_qa_chain(vector_store):
    try:
        # Get the retriever from vector store with increased k
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # Initialize the LLM
        llm = get_llm()
        if llm is None:
            st.error("Failed to initialize language model. Cannot continue.")
            return None

        # Initialize memory
        memory = setup_memory()

        # Create the Conversational QA chain with system prompt
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


# Function to process a user query
def process_query(query, qa_chain):
    try:
        with st.spinner("Searching for answer..."):
            # Use the updated invoke method instead of __call__
            result = qa_chain.invoke({"question": query})

            # Extract the answer and source documents
            answer = result.get("answer", "Sorry, I couldn't find an answer in the transcript.")
            source_docs = result.get("source_documents", [])

            # Log retrieval results
            logger.info(f"Retrieved {len(source_docs)} source documents for query: {query}")

            # Debug information about source documents
            if source_docs:
                for i, doc in enumerate(source_docs[:2]):  # Log first 2 docs for debugging
                    logger.info(f"Source doc {i}: {doc.page_content[:100]}...")
            else:
                logger.warning("No source documents retrieved!")
                # Add fallback behavior when no sources are retrieved
                if "I don't know" in answer or "I couldn't find" in answer or not answer.strip():
                    answer = "I couldn't find specific information about that in the video transcript. Could you try rephrasing your question or asking about another aspect of the video?"

            return answer, source_docs
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error(f"Error processing query: {str(e)}")
        return f"I encountered an error while processing your query: {str(e)}", []


# Main UI Layout
st.title("🎬 YouTube Video Q&A Bot")
st.markdown("Extract insights from YouTube videos by asking questions about their content.")

# Add a debug mode toggle
debug_mode = st.sidebar.toggle("Debug Mode", value=False)

# Input for YouTube URL
youtube_url = st.text_input("Enter YouTube URL:", key="url_input")

# Process button
if st.button("Process Video"):
    if youtube_url:
        try:
            # Reset chat history when processing a new video
            st.session_state.chat_history = []

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
                                st.success(
                                    f"Created embeddings and stored in Pinecone index under namespace: {video_namespace}")

                                # Set up QA chain
                                qa_chain = setup_qa_chain(vector_store)
                                if qa_chain:
                                    st.session_state.conversation = qa_chain
                                    st.success("Ready to answer questions about this video!")
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

# Display video info and transcription if available
if st.session_state.transcription and st.session_state.video_title:
    st.header("Video Information")
    st.subheader(st.session_state.video_title)
    st.markdown(f"Source: [{st.session_state.video_url}]({st.session_state.video_url})")

    with st.expander("Show Transcription"):
        st.write(st.session_state.transcription)

# Q&A Section - Chat Interface
if hasattr(st.session_state, 'conversation') and st.session_state.conversation:
    st.header("Chat with the Video")

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

    # Get user question
    user_question = st.chat_input("Ask a question about the video...")

    if user_question:
        # Add user message to chat UI
        st.chat_message("user").write(user_question)

        # Add to session state history
        st.session_state.chat_history.append(HumanMessage(content=user_question))

        # Get the answer
        answer, sources = process_query(user_question, st.session_state.conversation)

        # Display assistant response
        assistant_message = st.chat_message("assistant")
        assistant_message.write(answer)

        # Add to session state history
        st.session_state.chat_history.append(AIMessage(content=answer))

        # Display sources if available
        if sources:
            with assistant_message.expander("Sources from transcript"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i + 1}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
        elif debug_mode:
            st.warning("No sources were retrieved for this question.")

# Debug panel
if debug_mode:
    st.sidebar.header("Debug Information")

    if hasattr(st.session_state, 'vector_store') and st.session_state.vector_store:
        st.sidebar.subheader("Vector Store Info")
        st.sidebar.write(f"Index Name: {PINECONE_INDEX_NAME}")
        st.sidebar.write(f"Namespace: {st.session_state.video_namespace}")

        # Try to get namespace stats
        try:
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            if st.session_state.video_namespace in stats.get('namespaces', {}):
                vector_count = stats['namespaces'][st.session_state.video_namespace]['vector_count']
                st.sidebar.write(f"Vector Count: {vector_count}")
            else:
                st.sidebar.warning(f"Namespace {st.session_state.video_namespace} not found in index stats")
        except Exception as e:
            st.sidebar.error(f"Error getting index stats: {str(e)}")

    # Add log viewer
    st.sidebar.subheader("Logs")
    try:
        with open("youtube_qa_bot.log", "r") as log_file:
            logs = log_file.readlines()
            # Show last 10 log lines
            st.sidebar.code("".join(logs[-10:]))
    except:
        st.sidebar.info("No logs available yet")

# Helper section for API keys setup
with st.sidebar.expander("How to set up API keys"):
    st.markdown("""
    ### Setting up your .streamlit/secrets.toml file

    Create a `.streamlit` directory in your project folder and add a `secrets.toml` file with:

    ```toml
        # Required API keys
        PINECONE_API_KEY = "your-pinecone-api-key"
        PINECONE_ENVIRONMENT = "gcp-starter"  # Or your environment
        GOOGLE_API_KEY = "your-google-api-key"
        ```

        Get your API keys from:
        - [Pinecone Console](https://app.pinecone.io)
        - [Google AI Studio](https://makersuite.google.com/app/apikey)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; gap: 0.5rem; opacity: 0.8;">
    <p>Built with ❤️ using Streamlit, LangChain, Pinecone, and Google Gemini</p>
</div>
""", unsafe_allow_html=True)
