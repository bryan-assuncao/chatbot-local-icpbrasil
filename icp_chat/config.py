"""Configurações do assistente, lidas de variáveis de ambiente (ou de um arquivo .env)."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv é opcional
    pass

# Raiz do projeto: caminhos relativos são resolvidos a partir daqui,
# independentemente do diretório de onde o app é executado.
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    data_dir: Path
    persist_dir: Path
    collection_name: str
    ollama_url: str
    llm_model: str
    temperature: float
    request_timeout: float
    context_window: int
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def _path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else ROOT / path


def load_config() -> Config:
    return Config(
        data_dir=_path("ICP_DATA_DIR", "data"),
        persist_dir=_path("ICP_PERSIST_DIR", "chroma_db"),
        collection_name=os.getenv("ICP_COLLECTION", "icp_brasil_docs"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        llm_model=os.getenv("ICP_LLM_MODEL", "gemma3:12b"),
        temperature=float(os.getenv("ICP_TEMPERATURE", "0.0")),
        # Modelos locais podem demorar bastante na primeira resposta (carregamento em memória)
        request_timeout=float(os.getenv("ICP_REQUEST_TIMEOUT", "300")),
        # Enviado ao Ollama como num_ctx: precisa comportar o prompt + trechos recuperados + histórico
        context_window=int(os.getenv("ICP_CONTEXT_WINDOW", "8192")),
        embed_model=os.getenv("ICP_EMBED_MODEL", "intfloat/multilingual-e5-large"),
        # O multilingual-e5-large trunca entradas acima de 512 tokens; chunks maiores
        # teriam o final ignorado na busca vetorial.
        chunk_size=int(os.getenv("ICP_CHUNK_SIZE", "400")),
        chunk_overlap=int(os.getenv("ICP_CHUNK_OVERLAP", "60")),
        top_k=int(os.getenv("ICP_TOP_K", "8")),
    )
