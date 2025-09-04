# -*- coding: utf-8 -*-
"""
Agente IA HAMILTON RAMOS - Especialista em Segurança de Agroquímicos
Versão para desenvolvimento local e produção com Streamlit Community Cloud.
v2 - Código refatorado para ser à prova de cache de API Key.
"""

# --- 1. Importações de Bibliotecas ---
import streamlit as st
import os
import time
from dotenv import load_dotenv

# Importações específicas do LangChain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 2. Configurações Iniciais ---

# Carrega as variáveis de ambiente do ficheiro .env
load_dotenv()

# Definição de constantes para fácil manutenção
PDF_PATH = "manual-de-seguranca-na-aplicacao-de-agroquimicos-2021.pdf"
LOGO_PATH = "assets/Logo_IAC_20250519.jpeg" # Corrigido para minúsculas
OPENAI_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
MODEL_TEMPERATURE = 0.2

# --- 3. Funções Principais do Agente (com Cache) ---

@st.cache_resource
def get_llm_v2(): # <-- MUDANÇA 1: Nome da função alterado para forçar novo cache.
    """
    Cria e retorna uma instância do modelo de linguagem (LLM) da OpenAI.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Chave da API da OpenAI não encontrada. Verifique o seu ficheiro .env ou os Secrets no Streamlit Cloud.")
        return None
    try:
        return ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=api_key
        )
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo da OpenAI: {e}")
        return None

@st.cache_resource
def get_vector_store_from_pdf(pdf_path: str):
    """
    Processa o ficheiro PDF, cria os embeddings e armazena-os num vector store (FAISS).
    """
    api_key = os.getenv("OPENAI_API_KEY") # Pega a chave aqui também
    if not api_key:
        st.error("Chave da API da OpenAI não encontrada para os embeddings.")
        return None

    if not os.path.exists(pdf_path):
        st.error(f"Ficheiro PDF não encontrado em: {pdf_path}")
        return None
    try:
        with st.spinner(f"A processar o documento '{os.path.basename(pdf_path)}'..."):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # <-- MUDANÇA 2: Passando a chave de API explicitamente para os embeddings.
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=api_key)
            
            vector_store = FAISS.from_documents(splits, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Erro ao processar o PDF e criar o vector store: {e}")
        return None

def get_rag_chain(_llm, _retriever):
    """
    Cria e retorna a cadeia RAG completa que processa a pergunta e o histórico.
    """
    # ... (O resto desta função permanece exatamente igual) ...
    contextualize_q_system_prompt = (
        "Dada a seguinte conversa e uma pergunta de acompanhamento, reformule a pergunta de acompanhamento "
        "para ser uma pergunta independente, que possa ser entendida sem o histórico da conversa. "
        "NÃO responda à pergunta, apenas reformule-a se necessário; caso contrário, retorne-a como está."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=_llm, retriever=_retriever, prompt=contextualize_q_prompt
    )
    qa_system_prompt = """Você é a IA HAMILTON RAMOS, um assistente virtual especialista em segurança na aplicação de agroquímicos, baseado nos manuais e documentos fornecidos. A sua função é ajudar utilizadores a tirar dúvidas sobre as práticas seguras descritas nesses documentos.
    Use APENAS os seguintes trechos dos documentos recuperados para responder à pergunta. Seja claro, objetivo e direto ao ponto.
    Se a resposta não estiver nos trechos fornecidos, se a pergunta não for relacionada ao conteúdo dos documentos, ou se você não tiver certeza, diga educadamente que a informação específica não foi encontrada nos documentos fornecidos ou que não pode responder com base no contexto disponível. NÃO invente respostas.
    Mantenha a resposta concisa e focada na pergunta.
    Responda sempre em português do Brasil.

    Contexto Recuperado dos Documentos:
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(_llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 4. Interface Gráfica com Streamlit ---

st.set_page_config(page_title="Agente IA HAMILTON RAMOS", page_icon=LOGO_PATH)

st.image(LOGO_PATH, width=100)
st.title("Agente IA HAMILTON RAMOS.") # Ponto que adicionámos
st.caption("Especialista em segurança na aplicação de agroquímicos.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá! Sou a IA HAMILTON RAMOS. Como posso ajudar com as suas dúvidas sobre o manual de segurança?"),
    ]

# <-- MUDANÇA 3: Chamando a nova função renomeada.
llm = get_llm_v2()
vector_store = get_vector_store_from_pdf(PDF_PATH)

if llm and vector_store:
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 8})
    rag_chain = get_rag_chain(llm, retriever)
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("ai"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.markdown(message.content)

    user_query = st.chat_input("Digite a sua pergunta aqui...")

    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("human"):
            st.markdown(user_query)

        with st.chat_message("ai"):
            with st.spinner("A pensar..."):
                try:
                    start_time = time.time()
                    response = rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    end_time = time.time()
                    answer = response.get('answer', "Desculpe, não consegui gerar uma resposta.")
                    st.markdown(answer)
                    
                    sources = response.get('context', [])
                    if sources:
                        with st.expander(f"Fontes ({len(sources)} trechos consultados em {end_time - start_time:.2f}s)"):
                            for idx, doc in enumerate(sources):
                                source_file = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
                                page = doc.metadata.get('page', 'N/A')
                                content_preview = (doc.page_content[:200] + '...') if len(doc.page_content) > 200 else doc.page_content
                                st.info(f"Fonte {idx+1}: {source_file} (pág. {page + 1 if isinstance(page, int) else page})\n\"...{content_preview}...\"")
                    
                    st.session_state.chat_history.append(AIMessage(content=answer))

                except Exception as e:
                    error_message = f"Ocorreu um erro ao processar a sua pergunta: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append(AIMessage(content=error_message))
else:
    st.warning("O agente não pôde ser inicializado. Verifique as configurações e os ficheiros necessários.")








# # -*- coding: utf-8 -*-
# """
# Agente IA HAMILTON RAMOS - Especialista em Segurança de Agroquímicos
# Versão para desenvolvimento local e produção com Streamlit Community Cloud.
# """

# # --- 1. Importações de Bibliotecas ---
# import streamlit as st
# import os
# import time
# from dotenv import load_dotenv

# # Importações específicas do LangChain
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# # --- 2. Configurações Iniciais ---

# # Carrega as variáveis de ambiente do ficheiro .env
# load_dotenv()

# # Definição de constantes para fácil manutenção
# PDF_PATH = "manual-de-seguranca-na-aplicacao-de-agroquimicos-2021.pdf"
# LOGO_PATH = "assets/Logo_IAC_20250519.jpeg"
# OPENAI_MODEL_NAME = "gpt-4o-mini"
# EMBEDDING_MODEL_NAME = "text-embedding-3-small"
# MODEL_TEMPERATURE = 0.2

# # --- 3. Funções Principais do Agente (com Cache) ---

# @st.cache_resource
# def get_llm():
#     """
#     Cria e retorna uma instância do modelo de linguagem (LLM) da OpenAI.
#     A anotação @st.cache_resource garante que o modelo seja carregado apenas uma vez.
#     """
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         st.error("Chave da API da OpenAI não encontrada. Verifique o seu ficheiro .env.")
#         return None
#     try:
#         return ChatOpenAI(
#             model=OPENAI_MODEL_NAME,
#             temperature=MODEL_TEMPERATURE,
#             openai_api_key=api_key
#         )
#     except Exception as e:
#         st.error(f"Erro ao inicializar o modelo da OpenAI: {e}")
#         return None

# @st.cache_resource
# def get_vector_store_from_pdf(pdf_path: str):
#     """
#     Processa o ficheiro PDF, cria os embeddings e armazena-os num vector store (FAISS).
#     Esta função é intensiva e o cache é crucial para o desempenho.
#     """
#     if not os.path.exists(pdf_path):
#         st.error(f"Ficheiro PDF não encontrado em: {pdf_path}")
#         return None
#     try:
#         with st.spinner(f"A processar o documento '{os.path.basename(pdf_path)}'... Isto pode demorar um pouco na primeira execução."):
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
            
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             splits = text_splitter.split_documents(documents)
            
#             # Usando embeddings da OpenAI, otimizados para os seus modelos de chat
#             embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
            
#             vector_store = FAISS.from_documents(splits, embeddings)
#         return vector_store
#     except Exception as e:
#         st.error(f"Erro ao processar o PDF e criar o vector store: {e}")
#         return None

# def get_rag_chain(_llm, _retriever):
#     """
#     Cria e retorna a cadeia RAG completa que processa a pergunta e o histórico.
#     """
#     # 1. Cadeia para reformular a pergunta com base no histórico
#     contextualize_q_system_prompt = (
#         "Dada a seguinte conversa e uma pergunta de acompanhamento, reformule a pergunta de acompanhamento "
#         "para ser uma pergunta independente, que possa ser entendida sem o histórico da conversa. "
#         "NÃO responda à pergunta, apenas reformule-a se necessário; caso contrário, retorne-a como está."
#     )
#     contextualize_q_prompt = ChatPromptTemplate.from_messages([
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ])
#     history_aware_retriever = create_history_aware_retriever(
#         llm=_llm, retriever=_retriever, prompt=contextualize_q_prompt
#     )

#     # 2. Cadeia principal para responder à pergunta com base no contexto recuperado
#     qa_system_prompt = """Você é a IA HAMILTON RAMOS, um assistente virtual especialista em segurança na aplicação de agroquímicos, baseado nos manuais e documentos fornecidos. A sua função é ajudar utilizadores a tirar dúvidas sobre as práticas seguras descritas nesses documentos.
#     Use APENAS os seguintes trechos dos documentos recuperados para responder à pergunta. Seja claro, objetivo e direto ao ponto.
#     Se a resposta não estiver nos trechos fornecidos, se a pergunta não for relacionada ao conteúdo dos documentos, ou se você não tiver certeza, diga educadamente que a informação específica não foi encontrada nos documentos fornecidos ou que não pode responder com base no contexto disponível. NÃO invente respostas.
#     Mantenha a resposta concisa e focada na pergunta.
#     Responda sempre em português do Brasil.

#     Contexto Recuperado dos Documentos:
#     {context}"""
#     qa_prompt = ChatPromptTemplate.from_messages([
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ])
#     question_answer_chain = create_stuff_documents_chain(_llm, qa_prompt)

#     # 3. Combina as duas cadeias numa cadeia de recuperação final
#     return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# # --- 4. Interface Gráfica com Streamlit ---

# # Configuração da página
# st.set_page_config(page_title="Agente IA HAMILTON RAMOS", page_icon=LOGO_PATH)

# # Cabeçalho da aplicação
# st.image(LOGO_PATH, width=100)
# st.title("Agente IA HAMILTON RAMOS.")
# st.caption("Especialista em segurança na aplicação de agroquímicos.")

# # Inicialização do estado da sessão
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content="Olá! Sou a IA HAMILTON RAMOS. Como posso ajudar com as suas dúvidas sobre o manual de segurança?"),
#     ]

# # Carregamento do LLM e do Vector Store
# llm = get_llm()
# vector_store = get_vector_store_from_pdf(PDF_PATH)

# if llm and vector_store:
#     # Cria o retriever a partir do vector store
#     retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 8})
    
#     # Cria a cadeia RAG
#     rag_chain = get_rag_chain(llm, retriever)

#     # Exibição do histórico de chat
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("ai"):
#                 st.markdown(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("human"):
#                 st.markdown(message.content)

#     # Campo de entrada do utilizador
#     user_query = st.chat_input("Digite a sua pergunta aqui...")

#     if user_query:
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         with st.chat_message("human"):
#             st.markdown(user_query)

#         with st.chat_message("ai"):
#             with st.spinner("A pensar..."):
#                 try:
#                     start_time = time.time()
#                     response = rag_chain.invoke({
#                         "input": user_query,
#                         "chat_history": st.session_state.chat_history
#                     })
#                     end_time = time.time()
                    
#                     answer = response.get('answer', "Desculpe, não consegui gerar uma resposta.")
#                     st.markdown(answer)
                    
#                     # Exibe as fontes utilizadas para a resposta
#                     sources = response.get('context', [])
#                     if sources:
#                         with st.expander(f"Fontes ({len(sources)} trechos consultados em {end_time - start_time:.2f}s)"):
#                             for idx, doc in enumerate(sources):
#                                 source_file = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
#                                 page = doc.metadata.get('page', 'N/A')
#                                 content_preview = (doc.page_content[:200] + '...') if len(doc.page_content) > 200 else doc.page_content
#                                 st.info(f"Fonte {idx+1}: {source_file} (pág. {page + 1 if isinstance(page, int) else page})\n\"...{content_preview}...\"")
                    
#                     st.session_state.chat_history.append(AIMessage(content=answer))

#                 except Exception as e:
#                     error_message = f"Ocorreu um erro ao processar a sua pergunta: {e}"
#                     st.error(error_message)
#                     st.session_state.chat_history.append(AIMessage(content=error_message))
# else:
#     st.warning("O agente não pôde ser inicializado. Verifique as configurações e os ficheiros necessários.")