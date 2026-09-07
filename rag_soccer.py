# Importações básicas
from dotenv import load_dotenv
import os

# Loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader

# Divisão de texto em blocos
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Banco vetorial
from langchain_community.vectorstores import Chroma

# LLM
from langchain_openai import ChatOpenAI

# Cadeia RAG
from langchain.chains import RetrievalQA


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 1. Carregando o documento PDF e dividindo em blocos
soccer_file_path = "docs/soccer/regras_futebol.pdf"
documents = PyPDFLoader(soccer_file_path).load()
print(f"Documentos carregados: {len(documents)}")

# 2. Preparação dos documentos
# Os documentos precisam ser divididos em blocos para facilitar a recuperação do contexto.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
print(f"Tamanho dos chunks: {len(chunks)}")

# 3. Embedding e banco vetorial
# Cada bloco de texto será convertido em vetores semânticos e armazenado em um banco vetorial para recuperação eficiente.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

#Criando o banco vetorial com Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="docs/soccer/vectorstore"
)

# 4. Recuperando o contexto
# O retriever busca os trechos mais relevantes para cada pergunta feita pelo usuário.
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. Criando a cadeia RAG com LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

question = "Um jogador pode usar a mão para marcar um gol?"
response = qa_chain.invoke({"query": question})


print("Pergunta:")
print(question)

print("\nResposta do Agente:")
print(response["result"])

print("\nTrechos utilizados como contexto:")

for i, doc in enumerate(response["source_documents"], start=1):
    print(f"--- Trecho {i} ---")
    print(f"Fonte: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")
    print("Conteúdo:")
    print(doc.page_content)
    print("\n")