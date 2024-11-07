from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 임베딩 모델
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

text_path = './data/data.txt'

def text_data (text_path):
    # 텍스트 데이터 load, chunk
    loader = TextLoader(text_path, encoding='utf-8')
    data = loader.load()

    text_split = CharacterTextSplitter(
        separator='',
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len,
    )

    texts = text_split.split_text(data[0].page_content)

    # Vector로 저장
    vector_store = FAISS.from_texts(texts, embedding=embeddings_model)
    vector_store.save_local("faiss_index")

text_data(text_path)