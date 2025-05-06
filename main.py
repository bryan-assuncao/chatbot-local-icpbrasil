import chromadb
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# --- Configurações ---
persist_dir = "./chroma_db" # Diretório para salvar/carregar o índice persistente
collection_name = "icp_brasil_docs" # Nome da coleção no ChromaDB
data_directory = "./data/" # Diretório com os PDFs

# Configurar o LLM e os embeddings
llm = Ollama(
    model="gemma3:12b", # Modelo LLM a ser utilizado
    temperature=0.0, # Temperatura 0 para respostas mais objetivas e diretas
    request_timeout=80 # Aumentar o timeout para evitar erros de timeout
)
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large") 

Settings.llm = llm
Settings.embed_model = embed_model

# --- Lógica de Persistência ---
if not os.path.exists(persist_dir):
    print(f"Diretório de persistência '{persist_dir}' não encontrado.")
    print(f"Criando novo índice a partir dos documentos em '{data_directory}'...")

    # Verificar se o diretório de dados existe
    if not os.path.isdir(data_directory):
        print(f"Erro: O diretório de dados '{data_directory}' não foi encontrado.")
        exit()

    # Carregar documentos
    print(f"Carregando documentos PDF do diretório: {data_directory}")
    documents = SimpleDirectoryReader(
        input_dir=data_directory,
        required_exts=[".pdf"],
    ).load_data()

    if not documents:
        print(f"Nenhum documento PDF encontrado em '{data_directory}'.")
        exit()
    else:
        print(f"{len(documents)} documento(s) carregado(s) com sucesso.")

    # Configurar Chroma Persistente (Criação)
    print(f"Configurando ChromaDB persistente em '{persist_dir}'...")
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    # Usar get_or_create para segurança, embora neste fluxo ele criará
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Criar e persistir o índice
    print("Criando índice vetorial (pode levar tempo)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[SentenceSplitter(chunk_size=1500, chunk_overlap=250)], # Exemplo com chunk maior
        show_progress=True
    )
    # Não é necessário um comando 'save' explícito aqui, pois o StorageContext
    # já está vinculado ao ChromaVectorStore persistente. A indexação salva diretamente.
    print(f"Índice criado e salvo em '{persist_dir}'.")

else:
    # --- Carregar Índice Existente ---
    print(f"Carregando índice existente do diretório: '{persist_dir}'...")

    # Configurar Chroma Persistente (Carregamento)
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
    except Exception as e: # Tratar caso a coleção não exista por algum motivo
         print(f"Erro ao carregar coleção '{collection_name}' de '{persist_dir}': {e}")
         print("Verifique se o diretório não está corrompido ou tente recriar o índice apagando a pasta.")
         exit()

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Carregar o índice do armazenamento vetorial existente
    # Nota: Não precisamos do StorageContext tradicional para carregar assim
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model, # Ainda precisamos fornecer o embed_model
        # transformations não são necessários ao carregar, já foram aplicados
    )
    print("Índice carregado com sucesso.")


# --- Restante do Código (Query Engine e Loop) ---

# Template de resposta (sem alterações)
template = (
    """{context_str}

Você é um assistente jurídico especializado em certificação digital no âmbito do Instituto Nacional de Tecnologia da Informação (ITI) e da Infraestrutura de Chaves Públicas Brasileira (ICP-Brasil).

Seu papel é responder perguntas **exclusivamente com base nos documentos fornecidos no contexto acima**, sem citar ou se basear em outros documentos que não estejam expressamente incluídos.

Suas respostas devem ser:
- Claras, objetivas e completas;
- Rigorosamente baseadas no conteúdo disponível;
- Escritas em linguagem técnica, porém acessível.

⚠️ **Atenção:**
- **Não cite normas, leis, resoluções, instruções normativas ou documentos que não estejam presentes no contexto fornecido.**
- **Não mencione nomes de documentos, versões ou trechos que não estejam explicitamente presentes no conteúdo analisado.**
- **Nunca invente ou assuma informações, mesmo que pareçam plausíveis.**
- **Se a resposta envolver uma lista normativa (como requisitos, obrigações, procedimentos ou controles), enumere os itens com clareza, conforme descrito no conteúdo. Não omita pontos relevantes.**

Caso a pergunta não esteja contemplada no conteúdo fornecido, informe educadamente que a informação não está disponível e recomende consultar a legislação vigente diretamente no site do ITI ou com um profissional jurídico especializado em certificação digital.

Pergunta: {query_str}

Resposta:"""
)

qa_template = PromptTemplate(template)

query_engine = index.as_query_engine(
    text_qa_template=qa_template,
    similarity_top_k=5
)

# Loop de perguntas (sem alterações)
print("\nAssistente Jurídico ICP-Brasil (Persistente) - Pergunte algo ou digite 'sair' para encerrar.")
while True:
    user_input = input("\n Pergunta: ")
    if user_input.lower() in ["sair", "exit", "quit"]:
        print("Encerrando. Até mais!")
        break

    print("\nProcessando sua pergunta...")
    response = query_engine.query(user_input)
    print("\nResposta:\n", response.response)