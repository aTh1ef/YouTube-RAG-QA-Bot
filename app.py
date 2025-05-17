# app.py
import os
import streamlit as st
import tempfile
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import StdOutCallbackHandler

# Fix for LangSmithTracer import error
try:
    # Import LangSmith callback handler from the correct location
    from langchain.callbacks.tracers.langchain import LangChainTracer

    has_langsmith = True
except ImportError:
    has_langsmith = False


    # Fallback if not available
    class LangChainTracer:
        def __init__(self, *args, **kwargs):
            pass

# Updated Pinecone import for older version compatibility
import pinecone
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("langchain_processes.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("pdf_chatbot")


# Set up environment variables from Streamlit secrets
def setup_environment():
    # Set required API keys
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    # Set Pinecone environment for v2 compatibility
    if "PINECONE_ENVIRONMENT" in st.secrets:
        os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]

    # Set LangSmith variables if they exist in secrets
    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "pdf-chatbot")
        os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "true")
        logger.info("LangSmith tracing enabled")
    else:
        logger.info("LangSmith tracing not configured")


# Initialize environment
setup_environment()

# Initialize Pinecone with legacy v2 approach
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
)

# Initialize Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


class PDFChatbot:
    def __init__(self):
        logger.info("Initializing PDF Chatbot")

        # Initialize components first
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index_name = "pdf-chatbot"

        # Initialize callbacks list - use proper handlers
        callbacks = None  # Initially set to None

        # Only set up callbacks if LangSmith is properly configured
        if "LANGCHAIN_API_KEY" in os.environ and has_langsmith:
            try:
                tracer = LangChainTracer(
                    project_name=os.environ.get("LANGCHAIN_PROJECT", "pdf-chatbot")
                )
                callbacks = [StdOutCallbackHandler(), tracer]
                logger.info("LangSmith tracing enabled with callback handlers")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith tracing: {e}")
                callbacks = None

        # Initialize LLM with the updated model name for the free tier
        try:
            # First attempt using the latest model name format for Gemini
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            logger.info("Successfully initialized LLM with gemini-1.5-flash model")
        except Exception as e:
            logger.warning(f"Error initializing gemini-1.5-flash: {e}")
            try:
                # Fallback to another common model name
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro")
                logger.info("Successfully initialized LLM with gemini-1.0-pro model")
            except Exception as e2:
                logger.warning(f"Error initializing gemini-1.0-pro: {e2}")
                # Final fallback - configure a backup LLM
                self.llm = ChatGoogleGenerativeAI(model="models/gemini-pro")
                logger.info("Trying legacy model name format: models/gemini-pro")

        # Check if Pinecone index exists, create if not
        try:
            index_list = pinecone.list_indexes()
            if self.index_name not in index_list:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine"
                )
            logger.info(f"Using existing Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error with Pinecone index setup: {e}")
            st.error(f"Error with Pinecone index setup: {e}")

        # Initialize conversation memory - updated to address deprecation warning
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Add this to ensure proper memory retention
        )

        # Store callbacks for reuse - may be None if initialization failed
        self.callbacks = callbacks

        logger.info("PDF Chatbot initialization complete")

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        logger.info(f"Starting text extraction from PDF: {pdf_file.name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            pdf_path = tmp_file.name

        logger.info(f"Temporary PDF file created: {pdf_path}")

        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Extracted {len(documents)} pages from PDF")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            st.error(f"Error extracting text: {str(e)}")
            documents = []
        finally:
            # Clean up the temporary file
            os.unlink(pdf_path)
            logger.info(f"Temporary PDF file deleted")

        return documents

    def chunk_text(self, documents):
        """Split text into manageable chunks"""
        logger.info("Starting text chunking process")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks from documents")

        # Log first chunk as sample
        if chunks:
            logger.info(f"Sample chunk content: {chunks[0].page_content[:100]}...")

        return chunks

    def create_embeddings_and_store(self, chunks, namespace):
        """Create embeddings and store in Pinecone"""
        logger.info(f"Creating embeddings for {len(chunks)} chunks and storing in Pinecone namespace: {namespace}")

        try:
            # Get the index using legacy v2 approach
            index = pinecone.Index(self.index_name)
            
            # Delete any existing vectors in this namespace to avoid conflicts
            try:
                index.delete(deleteAll=True, namespace=namespace)
                logger.info(f"Deleted existing vectors in namespace: {namespace}")
            except Exception as e:
                logger.warning(f"No existing vectors to delete or error: {str(e)}")
                
            # Create the vector store with legacy approach
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=namespace
            )
            logger.info("Embeddings created and stored successfully")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            st.error(f"Error creating embeddings: {str(e)}")
            raise e

    def get_conversational_chain(self, vectorstore):
        """Create a conversational chain for Q&A"""
        logger.info("Creating conversational retrieval chain")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"  # Important to match with memory's output_key
        )

        logger.info("Conversational retrieval chain created")
        return chain

    def process_query(self, chain, query):
        """Process a user query and return a response"""
        logger.info(f"Processing query: {query}")

        result = chain({"question": query})

        logger.info(f"Retrieved {len(result['source_documents'])} source documents")
        logger.info(f"Generated answer (first 100 chars): {result['answer'][:100]}...")

        return result["answer"], result["source_documents"]


# Streamlit UI
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("AI PDF Chatbot")
st.write("Upload a PDF and chat with it!")

# Initialize session state
if "chatbot" not in st.session_state:
    # Use try/except to handle possible runtime errors during initialization
    try:
        st.session_state.chatbot = PDFChatbot()
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.session_state.chatbot = None

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Define the layout
left_col, right_col = st.columns([1, 2])

# Sidebar content
with left_col:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Add a debug mode toggle
    debug_mode = st.toggle("Debug Mode", value=False)

    if "LANGCHAIN_API_KEY" in st.secrets:
        langsmith_url = "https://smith.langchain.com/projects/" + st.secrets.get("LANGCHAIN_PROJECT", "pdf-chatbot")
        st.markdown(f"[View Traces in LangSmith]({langsmith_url})")

    if uploaded_file and not st.session_state.document_processed and st.session_state.chatbot:
        with st.spinner("Processing document..."):
            # Create a status container
            status_container = st.empty()

            try:
                # Extract text from PDF
                status_container.info("Extracting text from PDF...")
                documents = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)

                if not documents:
                    status_container.error("Failed to extract text from PDF. Please try another file.")
                else:
                    status_container.success(f"✅ Extracted {len(documents)} pages from PDF")

                    # Chunk text
                    status_container.info("Chunking text...")
                    chunks = st.session_state.chatbot.chunk_text(documents)
                    status_container.success(f"✅ Created {len(chunks)} text chunks")

                    # Create namespace from filename for organization in Pinecone
                    namespace = uploaded_file.name.replace(" ", "_").lower()

                    # Create embeddings and store in Pinecone
                    status_container.info("Creating embeddings and storing in Pinecone...")
                    vectorstore = st.session_state.chatbot.create_embeddings_and_store(chunks, namespace)
                    status_container.success("✅ Created embeddings and stored in Pinecone")

                    # Set up conversation chain
                    status_container.info("Setting up conversation chain...")
                    st.session_state.conversation = st.session_state.chatbot.get_conversational_chain(vectorstore)
                    st.session_state.document_processed = True
                    status_container.empty()
                    st.success("Document processed successfully!")
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                status_container.error(f"Error processing document: {str(e)}")

    # Display document stats if processed
    if st.session_state.document_processed:
        st.subheader("Document Information")
        st.info(f"Filename: {uploaded_file.name}")

        # Add reset button
        if st.button("Process Another Document"):
            st.session_state.document_processed = False
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.experimental_rerun()

# Chat interface
with right_col:
    st.header("Chat with your PDF")

    if not st.session_state.document_processed:
        st.info("Please upload a PDF document to start chatting.")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

        # User input
        user_query = st.chat_input("Ask a question about your document")

        if user_query:
            st.chat_message("user").write(user_query)

            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Get response from model
            with st.spinner("Thinking..."):
                try:
                    response, sources = st.session_state.chatbot.process_query(st.session_state.conversation,
                                                                               user_query)
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    response = f"I'm sorry, I encountered an error while processing your query: {str(e)}"
                    sources = []

            # Display assistant response
            st.chat_message("assistant").write(response)

            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Display sources (optional)
            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"Source {i + 1}:")
                        st.write(source.page_content)
                        st.write(f"Page: {source.metadata.get('page', 'N/A')}")
                        st.divider()

# Add a debug panel at the bottom of the page
if debug_mode:
    st.header("Debug Information")

    debug_tabs = st.tabs(["LangChain Components", "Process Flow", "Logs", "Pinecone Info"])

    with debug_tabs[0]:
        st.subheader("Document Loader")
        st.code("PyMuPDFLoader - Extracts text and metadata from PDF documents")

        st.subheader("Text Splitter")
        st.code("""
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
        """)

        st.subheader("Embedding Model")
        st.code("HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')")

        st.subheader("Vector Store")
        st.code(
            f"PineconeVectorStore(index_name='{st.session_state.chatbot.index_name if st.session_state.chatbot else 'pdf-chatbot'}')")

        st.subheader("Language Model")
        model_name = "gemini-1.5-flash (with fallbacks to gemini-1.0-pro or models/gemini-pro)"
        st.code(f"ChatGoogleGenerativeAI(model='{model_name}')")

        st.subheader("Chain Type")
        st.code("ConversationalRetrievalChain")

    with debug_tabs[3]:
        st.subheader("Pinecone Configuration")
        try:
            if st.session_state.chatbot:
                indexes = pinecone.list_indexes()
                st.write("Available Indexes:", indexes)
                st.write("Current Index:", st.session_state.chatbot.index_name)
                st.write("Environment:", os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter"))
                # Get pinecone client version
                st.write("Pinecone Client Version:", pinecone.__version__)
        except Exception as e:
            st.error(f"Error fetching Pinecone indexes: {str(e)}")

        st.subheader("Pinecone Information")
        st.info("""
        Pinecone Configuration:
        - Index dimensions: 384
        - Vector type: Dense
        - Metric: Cosine
        - Cloud Provider: AWS
        - Region: us-east-1
        - Environment: gcp-starter (default)
        """)

    with debug_tabs[1]:
        st.subheader("PDF Upload Process")
        st.markdown("""
        1. **PDF Upload** → User uploads PDF file
        2. **Text Extraction** → PyMuPDFLoader extracts text and metadata
        3. **Text Chunking** → RecursiveCharacterTextSplitter creates manageable chunks
        4. **Embedding Creation** → Each chunk converted to vector representation
        5. **Vector Storage** → Embeddings stored in Pinecone with namespace
        """)

        st.subheader("Query Process")
        st.markdown("""
        1. **Query Embedding** → User question converted to embedding
        2. **Vector Search** → Find similar document chunks in Pinecone
        3. **Context Retrieval** → Get top 5 most relevant chunks
        4. **Context + Question** → Combine with conversation history
        5. **LLM Generation** → Send to Gemini to generate response
        """)

    with debug_tabs[2]:
        try:
            with open("langchain_processes.log", "r") as log_file:
                logs = log_file.read()
            st.code(logs)
        except:
            st.info("No logs available yet")

# Create secrets.toml help
if not st.session_state.document_processed:
    with st.expander("How to set up API keys"):
        st.markdown("""
        ### Setting up your .streamlit/secrets.toml file

        Create a `.streamlit` directory in your project folder and add a `secrets.toml` file with the following content:

        ```toml
        # Required API keys
        PINECONE_API_KEY = "your-pinecone-api-key"
        GOOGLE_API_KEY = "your-google-api-key"

        # Required for Pinecone client v2.x.x
        PINECONE_ENVIRONMENT = "gcp-starter"  # Or your specific environment

        # Optional LangSmith configuration (for advanced monitoring)
        LANGCHAIN_API_KEY = "your-langsmith-api-key"
        LANGCHAIN_PROJECT = "pdf-chatbot"
        LANGCHAIN_TRACING_V2 = "true"
        ```

        You can get your API keys from:
        - [Pinecone Console](https://app.pinecone.io) (both API key and environment)
        - [Google AI Studio](https://makersuite.google.com/app/apikey)
        - [LangSmith](https://smith.langchain.com) (optional)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
    Built with ❤️ using <strong>Streamlit</strong>, <strong>LangChain</strong>, <strong>Pinecone</strong>, and <strong>Google Gemini</strong>
</div>
""", unsafe_allow_html=True)
