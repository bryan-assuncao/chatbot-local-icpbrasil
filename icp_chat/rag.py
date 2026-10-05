"""Núcleo RAG: indexação incremental dos PDFs no ChromaDB e motor de conversa."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import Config
from .prompts import CONDENSE_PROMPT, CONTEXT_PROMPT, SYSTEM_PROMPT

MANIFEST_FILE = "manifest.json"
# Incrementar sempre que a forma de gerar os trechos mudar, para forçar a reindexação completa
INDEX_SCHEMA_VERSION = 2
# Metadados visíveis para o LLM (os demais servem apenas para controle interno)
LLM_METADATA_KEYS = ("documento", "page_label")
EMBED_METADATA_KEYS = ("documento",)

Progress = Callable[[float, str], None]


def load_embed_model(model_name: str) -> BaseEmbedding:
    # Import tardio: carregar torch/transformers é lento
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Os modelos E5 foram treinados com os prefixos "query: " e "passage: ";
    # sem eles a qualidade da busca cai sensivelmente.
    is_e5 = "e5" in model_name.lower()
    return HuggingFaceEmbedding(
        model_name=model_name,
        query_instruction="query: " if is_e5 else None,
        text_instruction="passage: " if is_e5 else None,
    )


def make_llm(cfg: Config, model: str | None = None, temperature: float | None = None) -> Ollama:
    return Ollama(
        model=model or cfg.llm_model,
        base_url=cfg.ollama_url,
        temperature=cfg.temperature if temperature is None else temperature,
        request_timeout=cfg.request_timeout,
        context_window=cfg.context_window,
    )


def ollama_models(cfg: Config) -> list[str] | None:
    """Modelos disponíveis no Ollama, ou None se o servidor não estiver acessível."""
    try:
        with urllib.request.urlopen(f"{cfg.ollama_url.rstrip('/')}/api/tags", timeout=2) as resp:
            data = json.load(resp)
    except (OSError, ValueError):
        return None
    return sorted(m["name"] for m in data.get("models", []))


def document_title(path: Path) -> str:
    """'DOC ICP-01.01_v1.1.pdf' -> 'DOC-ICP-01.01'."""
    stem = path.stem.replace(" ", "-")
    return re.split(r"_v?\d", stem, maxsplit=1)[0]


def _clean_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rebuilt: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.rebuilt)

    def summary(self) -> str:
        if not self.changed:
            return "Índice já está atualizado."
        parts = [
            f"{len(self.added)} adicionado(s)",
            f"{len(self.updated)} atualizado(s)",
            f"{len(self.removed)} removido(s)",
        ]
        prefix = "Índice reconstruído: " if self.rebuilt else "Índice atualizado: "
        return prefix + ", ".join(parts) + "."


@dataclass
class Source:
    documento: str
    pagina: str
    score: float | None
    trecho: str


class KnowledgeBase:
    """Índice vetorial persistente dos PDFs, sincronizado de forma incremental.

    Um manifesto (chroma_db/manifest.json) guarda o hash de cada PDF indexado, de modo
    que apenas arquivos novos, alterados ou removidos são reprocessados.
    """

    def __init__(self, cfg: Config, embed_model: BaseEmbedding):
        self.cfg = cfg
        self.embed_model = embed_model
        cfg.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(cfg.persist_dir))
        self._manifest = self._read_manifest()
        self._hash_cache: dict[tuple[str, int, int], str] = {}
        # Índices criados por versões antigas ou com outra configuração de
        # embeddings/chunks são incompatíveis e precisam ser recriados.
        self.needs_rebuild = self._manifest.get("fingerprint") != self._fingerprint()
        self._open_collection()

    # --- Persistência ---------------------------------------------------------

    def _fingerprint(self) -> str:
        c = self.cfg
        return f"v{INDEX_SCHEMA_VERSION}|{c.embed_model}|{c.chunk_size}|{c.chunk_overlap}"

    def _read_manifest(self) -> dict:
        try:
            return json.loads((self.cfg.persist_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_manifest(self) -> None:
        path = self.cfg.persist_dir / MANIFEST_FILE
        path.write_text(json.dumps(self._manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _open_collection(self) -> None:
        self._collection = self._client.get_or_create_collection(
            self.cfg.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=self._collection),
            embed_model=self.embed_model,
        )

    # --- Sincronização --------------------------------------------------------

    def _file_hash(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._hash_cache:
            self._hash_cache[key] = _sha256(path)
        return self._hash_cache[key]

    def scan(self) -> dict[str, str]:
        """PDFs presentes em data_dir (inclusive subpastas): caminho relativo -> hash."""
        if not self.cfg.data_dir.is_dir():
            return {}
        return {
            path.relative_to(self.cfg.data_dir).as_posix(): self._file_hash(path)
            for path in sorted(self.cfg.data_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() == ".pdf"
        }

    def diff(self) -> tuple[dict[str, str], SyncReport]:
        current = self.scan()
        indexed = {} if self.needs_rebuild else self._manifest.get("files", {})
        report = SyncReport(
            added=sorted(current.keys() - indexed.keys()),
            updated=sorted(k for k in current.keys() & indexed.keys() if current[k] != indexed[k]),
            removed=sorted(indexed.keys() - current.keys()),
            rebuilt=self.needs_rebuild,
        )
        return current, report

    def sync(self, progress: Progress | None = None) -> SyncReport:
        progress = progress or (lambda fraction, message: None)
        current, report = self.diff()

        if report.rebuilt:
            try:
                self._client.delete_collection(self.cfg.collection_name)
            except Exception:  # coleção inexistente
                pass
            self._manifest = {"fingerprint": self._fingerprint(), "files": {}}
            self._write_manifest()
            self._open_collection()
            self.needs_rebuild = False

        files = self._manifest.setdefault("files", {})
        for rel in report.removed:
            self._collection.delete(where={"source": rel})
            files.pop(rel, None)
        self._write_manifest()

        splitter = SentenceSplitter(
            chunk_size=self.cfg.chunk_size, chunk_overlap=self.cfg.chunk_overlap
        )
        todo = report.added + report.updated
        for i, rel in enumerate(todo):
            progress(i / len(todo), f"Indexando {rel} ({i + 1}/{len(todo)})")
            # Remove trechos antigos (arquivo alterado ou indexação interrompida antes)
            self._collection.delete(where={"source": rel})
            nodes = splitter.get_nodes_from_documents(self._load_pdf(rel))
            self._index.insert_nodes(nodes)
            files[rel] = current[rel]
            # Salva a cada arquivo para que uma interrupção não perca o progresso
            self._write_manifest()
        if todo:
            progress(1.0, "Indexação concluída")
        return report

    def _load_pdf(self, rel: str) -> list[Document]:
        path = self.cfg.data_dir / rel
        title = document_title(path)
        documents = []
        for page in SimpleDirectoryReader(input_files=[path]).load_data():
            text = _clean_text(page.text)
            if not text:
                continue
            metadata = {
                "documento": title,
                "page_label": str(page.metadata.get("page_label", "?")),
                "file_name": path.name,
                "source": rel,
            }
            documents.append(
                Document(
                    text=text,
                    metadata=metadata,
                    excluded_llm_metadata_keys=[k for k in metadata if k not in LLM_METADATA_KEYS],
                    excluded_embed_metadata_keys=[k for k in metadata if k not in EMBED_METADATA_KEYS],
                )
            )
        return documents

    # --- Consulta -------------------------------------------------------------

    @property
    def indexed_files(self) -> list[str]:
        return sorted(self._manifest.get("files", {}))

    def chunk_count(self) -> int:
        return self._collection.count()

    def chat_engine(
        self, llm: LLM, top_k: int, chat_history: Sequence[ChatMessage] = ()
    ) -> CondensePlusContextChatEngine:
        """Motor de conversa: reescreve perguntas de acompanhamento com base no
        histórico, recupera os trechos relevantes e responde com base neles."""
        return CondensePlusContextChatEngine.from_defaults(
            retriever=self._index.as_retriever(similarity_top_k=top_k),
            llm=llm,
            chat_history=list(chat_history),
            system_prompt=SYSTEM_PROMPT,
            context_prompt=CONTEXT_PROMPT,
            condense_prompt=CONDENSE_PROMPT,
        )


def to_sources(nodes: Sequence[NodeWithScore]) -> list[Source]:
    sources = []
    for item in nodes:
        meta = item.node.metadata
        text = item.node.get_content().strip()
        sources.append(
            Source(
                documento=meta.get("documento") or meta.get("file_name", "?"),
                pagina=meta.get("page_label", "?"),
                score=item.score,
                trecho=text if len(text) <= 600 else text[:600].rstrip() + "…",
            )
        )
    return sources
