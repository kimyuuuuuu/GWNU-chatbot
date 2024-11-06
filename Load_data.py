from langchain_community.document_loaders import TextLoader
#from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


# TextLoader를 사용하여 텍스트 파일을 로드
loader = TextLoader('./data/data.txt', encoding='utf-8')
data = loader.load()  # data는 문자열을 담은 리스트입니다.

# 로드된 텍스트를 Document 객체 리스트로 변환
documents = [Document(page_content=text) for text in data]

# CharacterTextSplitter를 사용하여 텍스트를 나눕니다.
text_splitter = CharacterTextSplitter(
    separator='',
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

# 문서 내 텍스트를 분할하여 리스트에 저장
texts = []
for doc in documents:
    chunks = text_splitter.split_text(doc.page_content)  # 문자열로 분할
    texts.extend([Document(page_content=chunk) for chunk in chunks])  # 각 chunk를 Document로 저장

# OpenAI 임베딩을 사용하여 벡터 저장소 생성
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = FAISS.from_documents(texts, embedding=embeddings)
vector_store.save_local("faiss_index")