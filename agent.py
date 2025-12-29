import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "HVAC_Agent"

def ingest_pdf(pdf_path, project_type):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        
        for chunk in chunks:
            chunk.metadata["project"] = project_type
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        
        os.remove(pdf_path)
        
        return {"status": "success", "chunks": len(chunks)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def ask_question(question, project_type):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        results = db.similarity_search(question, k=3, filter={"project": project_type})
        
        if not results:
            return {"answer": "No information found", "confidence": "low"}
        
        context = "\n".join([r.page_content for r in results])
        return {"answer": context, "confidence": "high"}
    except Exception as e:
        return {"error": str(e)}

