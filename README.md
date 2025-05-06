# Assistente Jurídico ICP-Brasil com RAG e LLM Local

Este projeto implementa um sistema de Perguntas e Respostas (Q&A) especializado em documentos relacionados à Infraestrutura de Chaves Públicas Brasileira (ICP-Brasil). Ele utiliza um Modelo de Linguagem Grande (LLM) rodando localmente via Ollama (especificamente o modelo Gemma), técnicas de Retrieval Augmented Generation (RAG) com LlamaIndex, e armazenamento vetorial persistente com ChromaDB.

O objetivo é permitir que os usuários façam perguntas em linguagem natural sobre o conteúdo de documentos PDF específicos da ICP-Brasil e recebam respostas precisas, objetivas e estritamente baseadas nos textos fornecidos.

## Funcionalidades Principais

* **Carregamento de Documentos:** Lê arquivos PDF de um diretório especificado.
* **Processamento e Indexação:** Divide os documentos em partes (chunks), gera embeddings vetoriais usando o modelo `intfloat/multilingual-e5-large` da HuggingFace.
* **Armazenamento Persistente:** Salva e carrega o índice vetorial usando ChromaDB, evitando a necessidade de reprocessar os documentos a cada execução.
* **LLM Local:** Utiliza o Ollama para rodar o modelo de linguagem Gemma (ex: `gemma3:12b`) localmente, garantindo privacidade e controle.
* **Prompt Engineering:** Emprega um template de prompt customizado para instruir o LLM a responder exclusivamente com base no contexto fornecido e a seguir diretrizes específicas de linguagem técnica e objetividade.
* **Interface Interativa:** Permite que o usuário faça perguntas em um loop de console.
* **Robustez:** Inclui tratamento para verificar a existência do diretório de dados e do índice persistente.

## Como Funciona (Pipeline RAG)

1.  **Ingestão de Dados (Primeira Execução ou Atualização):**
    * Os documentos PDF no diretório `./data/` são carregados.
    * Os textos são divididos em chunks menores (`SentenceSplitter`).
    * Para cada chunk, um vetor de embedding é gerado pelo modelo `intfloat/multilingual-e5-large`.
    * Esses vetores e os textos correspondentes são armazenados em uma coleção no ChromaDB, persistida no diretório `./chroma_db/`.
2.  **Carregamento de Índice (Execuções Subsequentes):**
    * Se o diretório `./chroma_db/` existir, o índice vetorial é carregado diretamente do ChromaDB, poupando tempo de processamento.
3.  **Consulta do Usuário:**
    * O usuário insere uma pergunta.
    * A pergunta é convertida em um vetor de embedding usando o mesmo modelo.
    * O sistema busca no ChromaDB os chunks de texto cujos vetores são mais similares ao vetor da pergunta (`similarity_top_k`).
    * Esses chunks recuperados (o contexto) são combinados com a pergunta original e um prompt detalhado.
    * O LLM (Gemma via Ollama) recebe esse prompt enriquecido e gera uma resposta baseada *apenas* no contexto fornecido.

## Tecnologias Utilizadas

* **Python 3.x**
* **LlamaIndex:**
    * `llama-index-core`: Framework principal para RAG.
    * `llama-index-llms-ollama`: Integração com Ollama.
    * `llama-index-vector-stores-chroma`: Integração com ChromaDB.
    * `llama-index-embeddings-huggingface`: Para usar modelos de embedding da HuggingFace.
* **ChromaDB (`chromadb`):** Banco de dados vetorial para armazenamento e busca de embeddings.
* **Ollama:** Plataforma para rodar LLMs localmente.
    * **Modelo LLM:** `gemma3:12b` (ou outro modelo Gemma compatível).
* **HuggingFace Embeddings:**
    * **Modelo de Embedding:** `intfloat/multilingual-e5-large`.
* **Sentence Transformers (`sentence-transformers`):** Usado internamente pelo `HuggingFaceEmbedding`.

## Pré-requisitos

1.  **Python:** Versão 3.9 ou superior.
2.  **Ollama:**
    * Instale o Ollama: [https://ollama.com/](https://ollama.com/)
    * Baixe o modelo Gemma desejado. Exemplo para `gemma3:12b`:
        ```bash
        ollama pull gemma3:12b
        ```
    * Certifique-se de que o Ollama esteja rodando em segundo plano.
3.  **Git:** Para clonar o repositório.

## Configuração e Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <NOME_DO_SEU_REPOSITORIO>
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows:
    # venv\Scripts\activate
    # No macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Instale as dependências:**
    Crie um arquivo `requirements.txt` com o seguinte conteúdo:
    ```txt
    llama-index-core
    llama-index-llms-ollama
    llama-index-vector-stores-chroma
    llama-index-embeddings-huggingface
    chromadb
    sentence-transformers
    # Adicione outras dependências se houver (ex: pypdf se SimpleDirectoryReader precisar para PDFs)
    # llama-index-readers-file  # Para SimpleDirectoryReader, se não for pego automaticamente
    ```
    E então instale:
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: `llama-index-readers-file` pode ser necessário se `SimpleDirectoryReader` não conseguir ler PDFs por padrão. A LlamaIndex está modularizando seus pacotes.*

4.  **Prepare os Documentos:**
    * Crie um diretório chamado `data` na raiz do projeto.
    * Coloque os arquivos PDF da ICP-Brasil que você deseja consultar dentro deste diretório (`./data/`).

5.  **Verifique as Configurações no Script:**
    Abra o arquivo Python principal (ex: `seu_script.py`) e ajuste, se necessário:
    * `persist_dir = "./chroma_db"`: Diretório onde o índice do ChromaDB será salvo/carregado.
    * `collection_name = "icp_brasil_docs"`: Nome da coleção no ChromaDB.
    * `data_directory = "./data/"`: Diretório onde seus PDFs estão localizados.
    * `llm = Ollama(model="gemma3:12b", ...)`: Verifique se o nome do modelo corresponde ao que você baixou com Ollama.
    * `embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")`: Modelo de embedding.

## Como Usar

1.  **Execute o script Python:**
    ```bash
    python seu_script.py
    ```
    (Substitua `seu_script.py` pelo nome real do seu arquivo Python).

2.  **Primeira Execução:**
    * Se o diretório `./chroma_db` não existir, o script irá:
        * Carregar os documentos PDF do diretório `./data/`.
        * Processar os documentos, gerar embeddings e criar o índice no ChromaDB.
        * Este processo pode levar algum tempo, dependendo do número e tamanho dos documentos.
        * Você verá mensagens de progresso no console.

3.  **Execuções Subsequentes:**
    * Se o diretório `./chroma_db` já existir com um índice válido, o script irá carregá-lo rapidamente.

4.  **Interaja com o Assistente:**
    * Após a inicialização (criação ou carregamento do índice), você verá um prompt:
        ```
        Assistente Jurídico ICP-Brasil (Persistente) - Pergunte algo ou digite 'sair' para encerrar.

         Pergunta:
        ```
    * Digite sua pergunta sobre os documentos da ICP-Brasil e pressione Enter.
    * Aguarde o processamento e a resposta do assistente.
    * Para encerrar, digite `sair`, `exit` ou `quit`.

## Exemplo de Interação
