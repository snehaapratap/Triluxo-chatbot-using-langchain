import openai
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from url_loader import load_data_from_url

#place your api keys here
pinecone.init(api_key="", environment="")
openai.api_key = ""


def create_embeddings_and_store(url):
    documents = load_data_from_url(url)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()    
    index_name = "langchain-chatbot-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    
    vectorstore = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    print("Documents have been embedded and stored in the vector store.")
    
if __name__ == "__main__":
    url = "https://brainlox.com/courses/category/technical"
    create_embeddings_and_store(url)
