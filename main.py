"""Modo terminal do Assistente Jurídico ICP-Brasil.

Para a interface web, use:  streamlit run app.py
"""

from llama_index.core.llms import ChatMessage

from icp_chat import rag
from icp_chat.config import load_config


def main() -> None:
    cfg = load_config()
    print("Carregando modelo de embeddings...")
    kb = rag.KnowledgeBase(cfg, rag.load_embed_model(cfg.embed_model))

    report = kb.sync(progress=lambda fraction, message: print(f"  [{fraction:4.0%}] {message}"))
    print(report.summary())
    if kb.chunk_count() == 0:
        print(f"Nenhum documento PDF encontrado em '{cfg.data_dir}'.")
        return

    models = rag.ollama_models(cfg)
    if models is None:
        print(f"Aviso: Ollama não está acessível em {cfg.ollama_url}. Inicie-o com 'ollama serve'.")
    elif cfg.llm_model not in models:
        print(f"Aviso: modelo '{cfg.llm_model}' não encontrado. Execute 'ollama pull {cfg.llm_model}'.")

    llm = rag.make_llm(cfg)
    history: list[ChatMessage] = []
    print(
        f"\nAssistente Jurídico ICP-Brasil ({cfg.llm_model}) - "
        "digite 'limpar' para nova conversa ou 'sair' para encerrar."
    )
    while True:
        try:
            question = input("\nPergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            question = "sair"
        if not question:
            continue
        if question.lower() in ("sair", "exit", "quit"):
            print("\nEncerrando. Até mais!")
            break
        if question.lower() == "limpar":
            history.clear()
            print("Histórico da conversa apagado.")
            continue

        try:
            response = kb.chat_engine(llm, cfg.top_k, history).stream_chat(question)
            print("\nResposta:\n")
            answer = "".join(print(token, end="", flush=True) or token for token in response.response_gen)
            if getattr(response, "exception", None):
                raise response.exception
        except KeyboardInterrupt:
            print("\n(resposta interrompida)")
            continue
        except Exception as exc:
            print(f"\nErro ao gerar a resposta: {exc}")
            continue

        refs = dict.fromkeys(f"{s.documento}, p. {s.pagina}" for s in rag.to_sources(response.source_nodes))
        if refs:
            print("\n\nFontes: " + "; ".join(refs))
        history += [ChatMessage(role="user", content=question), ChatMessage(role="assistant", content=answer)]


if __name__ == "__main__":
    main()
