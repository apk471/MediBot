from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# print(PINECONE_API_KEY)


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
index_name = "medical"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)




#Creating Embeddings for Each of The Text Chunks & storing
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)