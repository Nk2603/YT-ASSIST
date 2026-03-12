import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate

# Load environment variables (e.g., GROQ_API_KEY, GOOGLE_API_KEY)
load_dotenv()

# --- Configuration ---
YOUTUBE_URL = "https://youtu.be/p4pHsuEf4Ms?t=9"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "openai/gpt-oss-120b"  # Note: Ensure this model name is correct for Groq
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

def format_docs(docs):
    """Formats a list of documents into a single string of concatenated page content."""
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_chain(youtube_url=None):
    """
    Initializes and returns the complete RAG (Retrieval-Augmented Generation) chain.
    """
    # 1. Load data from YouTube
    target_url = youtube_url if youtube_url else YOUTUBE_URL
    loader = YoutubeLoader.from_youtube_url(
        target_url,
        add_video_info=False,
        language="en"
    )
    documents = loader.load()

    # 2. Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Create Vector Store using Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Set up Retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 5. Define Prompt Template
    template = """
    You are an AI assistant. Use ONLY the provided context to answer the question.

    Context:
    {context}

    Question:
    {question}

    If the answer is not in the context, say:
    "I don't know based on the provided documents."

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # 6. Initialize LLM and Parser
    llm = ChatGroq(model=LLM_MODEL)
    parser = StrOutputParser()

    # 7. Construct the RAG Chain
    # We use RunnableParallel to handle context retrieval and question passing in parallel
    setup_and_retrieval = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    rag_chain = setup_and_retrieval | prompt | llm | parser
    return rag_chain

if __name__ == "__main__":
    # Initialize the RAG chain
    chain = initialize_rag_chain()

    # Execution Example
    question = "what are the topics covered in this video?"
    print(f"Question: {question}")
    
    result = chain.invoke(question)
    print("\nResult:")
    print(result)
