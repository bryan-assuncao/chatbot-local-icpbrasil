"""Interface web do Assistente Jurídico ICP-Brasil.

Execute com:  streamlit run app.py
"""

from dataclasses import asdict
from datetime import datetime

import streamlit as st
from llama_index.core.llms import ChatMessage

from icp_chat import rag
from icp_chat.config import load_config

st.set_page_config(page_title="Assistente ICP-Brasil", page_icon="🔐", layout="centered")

EXAMPLES = [
    "Quais são os tipos de certificado digital previstos na ICP-Brasil?",
    "Quais são as obrigações de uma Autoridade de Registro (AR)?",
    "Qual o prazo máximo de validade de um certificado do tipo A3?",
    "Como deve ser feita a identificação do titular na emissão de um certificado?",
]

cfg = load_config()
ss = st.session_state
ss.setdefault("messages", [])


@st.cache_resource(show_spinner="Carregando modelo de embeddings (a primeira execução pode demorar)…")
def get_knowledge_base() -> rag.KnowledgeBase:
    return rag.KnowledgeBase(cfg, rag.load_embed_model(cfg.embed_model))


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Fontes consultadas ({len(sources)})"):
        for s in sources:
            score = f" · similaridade {s['score']:.2f}" if s["score"] is not None else ""
            st.markdown(f"**{s['documento']}** — p. {s['pagina']}{score}")
            st.text(s["trecho"])


def conversation_markdown(messages: list[dict]) -> str:
    lines = [
        "# Conversa — Assistente ICP-Brasil",
        f"_Exportada em {datetime.now():%d/%m/%Y %H:%M}_",
        "",
    ]
    for m in messages:
        lines += [f"## {'Você' if m['role'] == 'user' else 'Assistente'}", "", m["content"], ""]
        refs = dict.fromkeys(f"{s['documento']}, p. {s['pagina']}" for s in m.get("sources", []))
        if refs:
            lines += ["**Fontes:** " + "; ".join(refs), ""]
    return "\n".join(lines)


def ask(question: str) -> None:
    ss.pending = question


kb = get_knowledge_base()

if "flash" in ss:
    st.toast(ss.pop("flash"), icon="✅")

# --- Barra lateral ------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Modelo")
    models = rag.ollama_models(cfg)
    if models is None:
        st.error(f"Ollama não está acessível em `{cfg.ollama_url}`. Inicie-o com `ollama serve`.")
        model = cfg.llm_model
    else:
        options = sorted(set(models) | {cfg.llm_model})
        model = st.selectbox("Modelo do Ollama", options, index=options.index(cfg.llm_model))
        if model not in models:
            st.warning(f"Modelo não baixado. Execute: `ollama pull {model}`")
    temperature = st.slider(
        "Temperatura", 0.0, 1.0, cfg.temperature, 0.05,
        help="Valores baixos geram respostas mais objetivas e fiéis aos documentos.",
    )
    top_k = st.slider(
        "Trechos recuperados", 2, 20, cfg.top_k,
        help="Quantidade de trechos dos documentos enviados ao modelo como contexto.",
    )

    st.header("📚 Base de documentos")
    _, pending = kb.diff()
    col1, col2 = st.columns(2)
    col1.metric("Documentos", len(kb.indexed_files))
    col2.metric("Trechos", kb.chunk_count())
    if pending.rebuilt:
        st.info("O índice precisa ser (re)criado com a configuração atual.")
    elif pending.changed:
        n = len(pending.added) + len(pending.updated) + len(pending.removed)
        st.info(f"{n} documento(s) com alterações pendentes de indexação.")
    if st.button("🔄 Atualizar índice", disabled=not pending.changed, width="stretch"):
        with st.status("Indexando documentos…", expanded=True) as status:
            bar = st.progress(0.0)
            report = kb.sync(lambda fraction, message: bar.progress(fraction, text=message))
            status.update(label=report.summary(), state="complete")
        ss.flash = report.summary()
        st.rerun()
    if kb.indexed_files:
        with st.expander("Documentos indexados"):
            st.markdown("\n".join(f"- {rag.document_title(cfg.data_dir / f)}" for f in kb.indexed_files))
    st.caption(f"Pasta de documentos: `{cfg.data_dir}`")

    st.header("💬 Conversa")
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Limpar", disabled=not ss.messages, width="stretch"):
        ss.messages = []
        st.rerun()
    col2.download_button(
        "⬇️ Exportar",
        conversation_markdown(ss.messages),
        file_name=f"conversa-icp-{datetime.now():%Y%m%d-%H%M}.md",
        mime="text/markdown",
        disabled=not ss.messages,
        width="stretch",
    )

# --- Conversa -----------------------------------------------------------------

st.title("🔐 Assistente ICP-Brasil")
st.caption(
    "Respostas baseadas exclusivamente nos documentos normativos (DOC-ICP) indexados. "
    "Tudo roda localmente: seus dados não saem da sua máquina."
)

ready = kb.chunk_count() > 0 and models is not None
if kb.chunk_count() == 0:
    st.warning("A base ainda não foi indexada. Clique em **Atualizar índice** na barra lateral.")

for message in ss.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

prompt = st.chat_input("Pergunte sobre a ICP-Brasil…", disabled=not ready) or ss.pop("pending", None)

if not ss.messages and not prompt:
    st.markdown("#### Experimente perguntar")
    for example in EXAMPLES:
        st.button(example, on_click=ask, args=(example,), disabled=not ready, width="stretch")

if prompt:
    history = [ChatMessage(role=m["role"], content=m["content"]) for m in ss.messages]
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando nos documentos…"):
                engine = kb.chat_engine(rag.make_llm(cfg, model, temperature), top_k, history)
                response = engine.stream_chat(prompt)
            answer = st.write_stream(response.response_gen)
            # Erros durante o streaming (ex.: modelo inexistente) ficam registrados na resposta
            if getattr(response, "exception", None):
                raise response.exception
            if not answer:
                raise RuntimeError("O modelo não retornou nenhuma resposta.")
        except Exception as exc:
            ss.messages.pop()
            st.error(f"Não foi possível gerar a resposta: {exc}")
            st.stop()
        sources = [asdict(s) for s in rag.to_sources(response.source_nodes)]
        render_sources(sources)

    ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()
