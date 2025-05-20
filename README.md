# 🎬 YouTube RAG AI Assistant

A conversational AI application that lets users ask questions about YouTube video content using advanced RAG (Retrieval Augmented Generation) techniques. The app extracts transcripts from YouTube videos, processes them, and enables natural language question-answering about the video content.

🌐 Live Demo:
Try out the live demo of this application at:
[https://ai-pdf-langchain-chatbot-app.streamlit.app/](https://youtube-rag-ai-bot.streamlit.app)


## 📋 Features

- Extract transcripts directly from any YouTube video with captions
- Process and index video content using state-of-the-art vector embeddings
- Ask natural language questions about the video content
- Get AI-generated answers with source references from the transcript
- Maintain conversational context for follow-up questions
- Debug mode for troubleshooting and development

## 🔧 Technology Stack

- **Frontend & Application**: [Streamlit](https://streamlit.io/) - For creating the web interface
- **Language Models**: [Google Gemini](https://ai.google.dev/) - For natural language understanding and generation
- **Vector Database**: [Pinecone](https://www.pinecone.io/) - For efficient similarity search and retrieval
- **Embedding Models**: [Hugging Face](https://huggingface.co/) - Sentence transformers for text embeddings
- **RAG Framework**: [LangChain](https://python.langchain.com/) - For building the retrieval-augmented generation pipeline
- **YouTube Integration**: [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) - For extracting video transcripts
- **Debugging & Tracing**: [LangSmith](https://smith.langchain.com/) - For monitoring and debugging LangChain applications

## 🔄 How It Works: RAG Workflow

1. **Transcript Extraction**:
   - User inputs a YouTube URL
   - The application extracts the video ID and fetches the transcript using YouTube Transcript API
   - Video title is also retrieved for reference

2. **Document Processing**:
   - The transcript is split into smaller chunks using LangChain's RecursiveCharacterTextSplitter
   - Each chunk is processed and metadata (video title, chunk ID) is added

3. **Vector Embedding & Storage**:
   - Text chunks are converted to vector embeddings using HuggingFace's sentence transformers model
   - Embeddings are stored in Pinecone vector database under a namespace derived from the video title
   - Each video gets its own namespace to keep content separate

4. **Retrieval Setup**:
   - A vector retrieval system is initialized to find relevant text chunks based on questions
   - The retriever is configured to fetch the most relevant chunks (k=5)

5. **Question Answering**:
   - User asks questions about the video content
   - The question is converted to an embedding and used to search for similar chunks
   - The most relevant chunks are retrieved from Pinecone
   - Google Gemini processes the retrieved context plus the question to generate an answer
   - The application shows both the answer and the source transcript segments

6. **Conversation Memory**:
   - The system maintains a conversation history
   - This enables follow-up questions and maintaining context

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pinecone API key
- Google AI Studio API key
- YouTube video URLs with available captions

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/youtube-rag-assistant.git
   cd youtube-rag-assistant
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.streamlit` directory and set up your API keys:
   ```bash
   mkdir -p .streamlit
   touch .streamlit/secrets.toml
   ```

4. Edit the `secrets.toml` file with your API keys:
   ```toml
   # Required API keys
   HUGGING_FACE_API_KEY = "your-huggingface-api-key"
   PINECONE_API_KEY = "your-pinecone-api-key"
   PINECONE_ENVIRONMENT = "your-pinecone-environment"  # The environment from your Pinecone host URL
   GOOGLE_API_KEY = "your-google-api-key"
   
   # LangSmith Configuration (Optional - for tracing and debugging)
   LANGCHAIN_API_KEY = "your-langsmith-api-key"
   LANGCHAIN_PROJECT = "youtube-qa-bot"
   LANGCHAIN_TRACING_V2 = "true"
   ```

### Getting API Keys

- **Hugging Face API Key**:
  1. Create an account on [Hugging Face](https://huggingface.co/)
  2. Go to your profile settings and navigate to "Access Tokens"
  3. Create a new token with "read" access

- **Pinecone API Key**:
  1. Create an account on [Pinecone](https://app.pinecone.io/)
  2. Create a project and get your API key from the dashboard
  3. Note your environment (e.g., "gcp-starter") from your Pinecone host URL

- **Google Gemini API Key**:
  1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
  2. Create or use an existing project
  3. Generate an API key for Gemini models

- **LangSmith API Key (Optional - for tracing and debugging)**:
  1. Sign up for [LangSmith](https://smith.langchain.com/)
  2. Create a new project called "youtube-qa-bot"
  3. Generate an API key from your settings
  4. LangSmith provides tracing and debugging tools for LangChain applications

### Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

## 📖 Usage Guide

1. Enter a YouTube URL in the text input field
2. Click "Process Video" to extract and process the transcript
3. Once processing is complete, you can ask questions about the video content
4. The application will display answers based on the video transcript
5. You can ask follow-up questions and maintain a conversation about the video

## 🔍 Debug Mode

### In-App Debugging

Toggle "Debug Mode" in the sidebar to see additional information:
- Vector store details
- Namespace statistics
- Recent log entries
- Source documents for answers

### LangSmith Tracing (Optional)

If you've configured LangSmith:
- All LangChain executions are traced and logged to your LangSmith project
- View detailed execution traces, input/output pairs, and performance metrics
- Debug complex chains and identify bottlenecks in your RAG pipeline
- Visualize how your application processes questions and generates answers

## 🛠️ Common Issues and Troubleshooting

- **No transcript available error**: Some YouTube videos don't have captions or have disabled transcripts. Try another video.
- **API key errors**: Double-check your API keys in the `.streamlit/secrets.toml` file.
- **Embedding model errors**: Ensure you have enough RAM to load the embedding model. Consider using a lighter model if needed.
- **Pinecone connection issues**: Verify your Pinecone environment and API key are correct.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Uses [LangChain](https://python.langchain.com/) for RAG pipeline
- Powered by [Google Gemini](https://ai.google.dev/) for AI responses
- Vector search by [Pinecone](https://www.pinecone.io/)
- Transcript extraction via [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- Debugging and tracing with [LangSmith](https://smith.langchain.com/)
