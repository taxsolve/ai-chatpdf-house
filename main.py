__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # 수정
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("PDF파일을 올려주세요", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages
    
#.업로드가 되면 동작하는 코드드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    # Embedding

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    #load Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Qustions
    st.header("PDF에게 질문해보세요")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        with st.spinner("질문을 처리하는 중입니다..."):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])



