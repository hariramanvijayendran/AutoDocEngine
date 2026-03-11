"""
Entry point — starts the Autonomous Document Workflow Engine.
Run with:  python run.py
"""
import uvicorn
from engine.config import API_HOST, API_PORT

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║   Autonomous Document Workflow Engine  v1.0          ║
║   Event-Driven Agents + RAG                          ║
╠══════════════════════════════════════════════════════╣
║  Dashboard →  http://localhost:8000                  ║
║  API Docs  →  http://localhost:8000/docs             ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
