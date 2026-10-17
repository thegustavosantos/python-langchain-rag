import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

FILES = [
    "docs/human_resources/politica_ferias.pdf",
    "docs/human_resources/politica_home_office.pdf",
    "docs/human_resources/codigo_conduta.pdf"
]
PERSIST_DIRECTORY="docs/human_resources/vectorstore"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini"

@st.cache_data
def load_documents():
    documents = []
    for file in FILES:
        loader = PyPDFLoader(file)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file
        documents.extend(docs)
    return documents;

def generate_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def enrich_chunks_with_metadata(chunks):
    for chunk in chunks:
        text = chunk.page_content.lower()
        if "ferias" in text or "férias" in text:
            chunk.metadata["category"] = "Vacation"
        elif "home office" in text or "remoto" in text:
            chunk.metadata["category"] = "Remote"
        elif "conduta" in text or "etica" in text or "ética" in text:
            chunk.metadata["category"] = "Conduct"
        else:
            chunk.metadata["category"] = "General"
    return chunks

@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

def rerank_documents(question, documents, llm):
    """
    Reordena os documentos recuperados com base na relevância
    usando o próprio LLM (reranking semântico)
    """

    prompt_rerank = PromptTemplate(
        input_variables=["question", "text"],
        template="""Dê uma nota de relevância de 0 a 10 para o trecho em relação à pergunta.
Considere 0 irrelevante e 10 altamente relevante. Responda somente com o número.

Pergunta: {question}
Trecho: {text}
Nota:"""
    )

    documents_with_score = []
    for doc in documents:
        score = llm.invoke(
            prompt_rerank.format(
                question=question,
                text=doc.page_content
            )
        ).content.strip()

        try:
            score = float(score)
        except:
            score = 0.0
        documents_with_score.append((score, doc))

    documents_sorted_by_score = sorted(
        documents_with_score,
        key=lambda x: x[0],
        reverse=True
    )
    return [doc for _, doc in documents_sorted_by_score]

def create_qa_chain(question, vectorstore):
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME, 
        api_key=api_key, 
        temperature=0.0
    )

    documents_retrieved = vectorstore.similarity_search(
        question, 
        k=8
    )

    documents_reranked = rerank_documents(question, documents_retrieved, llm)
    best_documents = documents_reranked[:4]

    context = "\n\n".join([doc.page_content for doc in best_documents])

    prompt_final = f"""Você é um assistente de Recursos Humanos. Responda em português usando somente o contexto abaixo.
Se o contexto não contiver a resposta, diga que não encontrou essa informação nos documentos.

Contexto:
{context}

Pergunta: {question}
Resposta:"""
    answer = llm.invoke(prompt_final);
    return answer.content, best_documents


st.title("Assistente de Recursos Humanos")

with st.form("question_form"):
    question = st.text_input("Sua pergunta")
    submitted = st.form_submit_button("Perguntar")

if submitted:
    if not question.strip():
        st.warning("Digite uma pergunta.")
    elif not api_key:
        st.error("A variável OPENAI_API_KEY não foi encontrada no arquivo .env.")
    else:
        try:
            with st.spinner("Consultando os documentos..."):
                documents = load_documents()
                chunks = generate_chunks(documents)
                enriched_chunks = enrich_chunks_with_metadata(chunks)
                vectorstore = create_vectorstore(enriched_chunks)
                answer, source_documents = create_qa_chain(question, vectorstore)

            st.markdown(answer)
            with st.expander("Fontes consultadas"):
                for document in source_documents:
                    source = document.metadata.get("source", "Documento")
                    page = document.metadata.get("page")
                    label = f"{source} - página {page + 1}" if isinstance(page, int) else source
                    st.caption(label)
                    st.write(document.page_content)
        except Exception as error:
            st.error(f"Não foi possível responder à pergunta: {error}")
