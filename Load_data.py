from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

loader = TextLoader('./data/data.txt', encoding='utf-8')
data = loader.load()

text_split = CharacterTextSplitter(
    separator='',
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
)

texts = text_split.split_text(data[0].page_content)

vector_store = FAISS.from_texts(texts, embedding=embeddings_model)
vector_store.save_local("faiss_index")
