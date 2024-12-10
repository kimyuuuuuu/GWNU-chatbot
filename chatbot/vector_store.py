# HugoingFace Embeddings를 다운로드
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


#texts = './data/all_text_data.txt'
texts = './data/only_text_data.txt'

# embeddings_model = HuggingFaceEmbeddings(
#     model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
# )

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

 # Vector로 저장
vector_store = FAISS.from_texts(texts, embedding=embeddings_model)
#vector_store.save_local("./chatbot/all_faiss_index")
vector_store.save_local("./chatbot/only_text_faiss_index")