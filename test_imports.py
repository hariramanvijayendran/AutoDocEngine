"""Targeted import test."""
import sys
print("Python:", sys.version[:10])

def check(label, code):
    try:
        exec(code)
        print(f"  ✅ {label}")
    except Exception as e:
        print(f"  ❌ {label}: {type(e).__name__}: {e}")

check("langchain_core.messages",  "from langchain_core.messages import SystemMessage, HumanMessage")
check("langchain_core.prompts",   "from langchain_core.prompts import ChatPromptTemplate")
check("langchain_core.parsers",   "from langchain_core.output_parsers import StrOutputParser")
check("langchain_ollama ChatOllama", "from langchain_ollama import ChatOllama")
check("langchain_community.vectorstores.FAISS", "from langchain_community.vectorstores import FAISS")
check("langchain_ollama OllamaEmbeddings", "from langchain_ollama import OllamaEmbeddings")
check("ingestion_agent",   "from engine.agents.ingestion_agent import ingestion_agent")
check("classif_agent",     "from engine.agents.classification_agent import classification_agent")
check("extraction_agent",  "from engine.agents.extraction_agent import extraction_agent")
check("summary_agent",     "from engine.agents.summary_agent import summary_agent")
check("router_agent",      "from engine.agents.router_agent import router_agent")
check("notif_agent",       "from engine.agents.notification_agent import notification_agent")
check("indexer",           "from engine.rag.indexer import index_document")
check("retriever",         "from engine.rag.retriever import retrieve")
check("pipeline",          "from engine.pipeline import run_pipeline")
print("All done.")
