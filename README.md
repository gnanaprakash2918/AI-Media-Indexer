# AI-Media-Indexer
AI engine for searching and navigating large media libraries using text, visuals, and audio.

- Right now, The Goal is to write a Microservices-inspired Modular Monolith for this project.
- Tech Stack Decided :
  - Orchestrator : Google ADK (Agent Development Kit) to build the necessary agents.
  - Ingestion Engine : Python with FFMPEG (For Media processing)
  - LLM : Plans on using Ollama with locally hosted models. But Also providing the ability to add other LLMs as well.
  - Memory : Using Qdrant with Docker for Vector Store.
  - Backend : FastAPI for the backend server.