from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorRag:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " "]
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name = embedding_model_name,
            model_kwargs = {
                'trust_remote_code': True
            },
            encode_kwargs = {
                'normalize_embeddings': True
            }
        )

    def build_vector_index(self, resume_text: str):
        chunks = self.text_splitter.split_text(resume_text)
        docs = [Document(page_content = c, metadata = {'chunk_id': i}) 
                for i, c in enumerate(chunks)]
        return FAISS.from_documents(
            documents=docs, 
            embedding=self.embeddings
        )
    
    def retrieve_relevant_chunks(self, query: str, vector_store, k: int = 5):
        results = vector_store.similarity_search(query, k = k)
        return [doc.page_content for doc in results]


        