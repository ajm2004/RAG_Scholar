import os
import logging
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pickle

from langchain.document_loaders import WikipediaLoader, ArxivLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub, OpenAI

from langchain.prompts import PromptTemplate
import streamlit as st

from huggingface_hub import InferenceClient, __version__

from dotenv import load_dotenv
import os

# RAG components

# UI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        # Import inside the method to prevent global side effects
        import os
        import torch
        from sentence_transformers import SentenceTransformer
        
        # Configure the environment
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Try to load the model with safeguards
        try:
            # First attempt: standard approach
            self.model = SentenceTransformer(model_name)
        except NotImplementedError:
            # Second attempt: force CPU before loading
            torch.set_default_tensor_type(torch.FloatTensor)
            self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

class RAGScholar:
    def __init__(self, source_type="wikipedia", topic=None, model_name="google/flan-t5-base", 
             use_openai=False, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG Scholar
        
        Args:
            source_type: Either "wikipedia" or "arxiv"
            topic: Topic or category to focus on
            model_name: HuggingFace model name for LLM
            use_openai: Whether to use OpenAI API
            embedding_model: Model name for embeddings
        """
        self.source_type = source_type
        self.topic = topic
        self.model_name = model_name
        self.use_openai = use_openai
        self.embedding_model = embedding_model
        
        # Initialize components
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # Use your custom embeddings class instead of HuggingFaceEmbeddings
        self.embeddings = CustomSentenceTransformerEmbeddings(embedding_model)
        
        # Setup LLM
        if use_openai:
            self.llm = OpenAI(temperature=0.2)
        else:
            # Use the appropriate model class for T5
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
            import torch
            
            # Different model loading based on model type
            model_id = self.model_name
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Check if it's a T5-based model
            if "t5" in model_id.lower():
                # T5 is a seq2seq model
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                
                # Move model to device after initialization
                model = model.to("cpu")  # or "cuda" if you have GPU
                
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.2
                )
            else:
                # For causal LMs like Falcon
                from transformers import AutoModelForCausalLM
                
                model = AutoModelForCausalLM.from_pretrained(model_id)
                
                # Move model to device after initialization
                model = model.to("cpu")  # or "cuda" if you have GPU
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.2
                )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)

            
        # Setup vector store and QA chain
        self.vectorstore = None
        self.qa_chain = None
        
        # Data paths
        self.db_path = Path(f"data/vector_db_{self.source_type}_{self.topic}")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
    
    def load_data(self, limit=100):
        """Load data from the selected source"""
        logger.info(f"Loading data from {self.source_type} on topic: {self.topic}")
        
        if self.source_type == "wikipedia":
            loader = WikipediaLoader(query=self.topic, load_max_docs=limit)
            self.documents = loader.load()
        else:  # arxiv
            # For ArXiv, we need to formulate a query based on the topic
            query = f"ti:{self.topic}" if self.topic else "cat:cs.AI"  # Default to AI if no topic
            loader = ArxivLoader(query=query, load_max_docs=limit)
            self.documents = loader.load()
        
        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents
        
    def process_documents(self):
        """Split documents into chunks and create embeddings"""
        logger.info("Processing documents...")
        
        # Split documents into chunks
        self.chunks = self.text_splitter.split_documents(self.documents)
        logger.info(f"Split into {len(self.chunks)} chunks")
        
        # Create and save vectorstore
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        if not self.db_path.exists():
            self.vectorstore.save_local(self.db_path)
            logger.info(f"Vector database saved to {self.db_path}")
        
        return self.chunks
    
    def load_vectorstore(self):
        if self.db_path.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vectorstore loaded from disk")
                return True
            except (ValueError, KeyError, AttributeError, pickle.UnpicklingError) as e:
                logger.warning(f"Vectorstore incompatible ({e}); rebuilding…")
                return False
        return False
    
    
    def setup_qa_chain(self, k=4):
        """Set up the QA chain with the vectorstore"""
        # Create a custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer: """
        
        QA_PROMPT = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        
        logger.info("QA chain setup complete")
    
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        """
        Answer a question using the RAG pipeline
        
        Args:
            question: The question to answer
            
        Returns:
            Tuple containing the answer and list of source passages
        """
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
            
        logger.info(f"Answering question: {question}")
        
        # Get answer from QA chain
        result = self.qa_chain({"query": question})
        
        # Extract answer and sources
        answer = result["result"]
        source_documents = result["source_documents"]
        source_passages = [doc.page_content for doc in source_documents]
        
        return answer, source_passages
    
    def initialize(self, force_reload=False):
        """Initialize the RAG pipeline"""
        # Try to load existing vectorstore
        if not force_reload and self.load_vectorstore():
            logger.info("Using existing vector database")
        else:
            # Load and process documents
            self.load_data()
            self.process_documents()
        
        # Setup QA chain
        self.setup_qa_chain()
        
        logger.info("RAG Scholar initialized successfully")


# Streamlit UI code
def create_ui():
    st.set_page_config(page_title="RAG Scholar", page_icon="🧠", layout="wide")
    
    st.title("🧠 RAG Scholar")
    st.subheader("A Research Assistant for Wikipedia or ArXiv Topics")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        source_type = st.radio("Select Data Source", ["wikipedia", "arxiv"])
        topic = st.text_input("Topic/Category", value="machine learning")
        use_openai = st.checkbox("Use OpenAI (requires API key)", value=False)
        
        model_options = {
            "OpenAI": "gpt-3.5-turbo",
            "Flan-T5": "google/flan-t5-base",
            "Falcon": "tiiuae/falcon-7b-instruct",
        }
        
        if use_openai:
            model_name = model_options["OpenAI"]
        else:
            model_select = st.selectbox("Select Model", ["Flan-T5", "Falcon"])
            model_name = model_options[model_select]
        
        init_button = st.button("Initialize/Reload Knowledge Base")
    
    # Main area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("Ask a Question")
        question = st.text_input("Enter your question")
        submit_button = st.button("Submit")
    
    # Initialize session state
    if "rag_scholar" not in st.session_state or init_button:
        with st.spinner("Initializing knowledge base..."):
            st.session_state["rag_scholar"] = RAGScholar(
                source_type=source_type,
                topic=topic,
                model_name=model_name,
                use_openai=use_openai
            )
            st.session_state["rag_scholar"].initialize(force_reload=init_button)
        st.success(f"Knowledge base initialized with {source_type} data on {topic}!")
    
    if submit_button and question:
        with st.spinner("Thinking..."):
            answer, sources = st.session_state["rag_scholar"].answer_question(question)
            
        with col1:
            st.subheader("Answer")
            st.write(answer)
        
        with col2:
            st.subheader("Source Passages")
            for i, source in enumerate(sources):
                with st.expander(f"Source {i+1}"):
                    st.write(source)

# Command line interface
def main():
    parser = argparse.ArgumentParser(description="RAG Scholar - Research Assistant")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit UI")
    parser.add_argument("--source", choices=["wikipedia", "arxiv"], default="wikipedia", help="Data source")
    parser.add_argument("--topic", default="machine learning", help="Topic or category")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API")
    parser.add_argument("--question", help="Question to answer in CLI mode")
    
    args = parser.parse_args()
    
    if args.ui:
        # Launch Streamlit UI
        os.system("streamlit run app.py")
    else:
        # CLI mode
        rag = RAGScholar(source_type=args.source, topic=args.topic, use_openai=args.openai)
        rag.initialize()
        
        if args.question:
            answer, sources = rag.answer_question(args.question)
            print("\n=== Answer ===")
            print(answer)
            print("\n=== Sources ===")
            for i, source in enumerate(sources):
                print(f"\nSource {i+1}:")
                print(source[:300] + "..." if len(source) > 300 else source)
        else:
            # Interactive mode
            print(f"RAG Scholar initialized with {args.source} data on '{args.topic}'")
            while True:
                question = input("\nEnter a question (or 'exit' to quit): ")
                if question.lower() == 'exit':
                    break
                
                answer, sources = rag.answer_question(question)
                print("\n=== Answer ===")
                print(answer)
                print("\n=== Sources ===")
                for i, source in enumerate(sources):
                    print(f"\nSource {i+1}:")
                    print(source[:300] + "..." if len(source) > 300 else source)

if __name__ == "__main__":
    # In Jupyter Notebook, we don't have __file__
    # Check if we're running in a notebook

    # This will only work in a regular Python script
    if Path(__file__).name == "app.py":
        # Running as Streamlit app
        create_ui()
    else:
        # Running as CLI
        main()
