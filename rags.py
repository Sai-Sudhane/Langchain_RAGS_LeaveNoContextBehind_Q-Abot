from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import NLTKTextSplitter
from langchain_chroma import Chroma


# Loading the pdf file
loader = PyPDFLoader("pdf/Leave_no_context_behind.pdf")
pages = loader.load()

text_spliter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_spliter.split_documents(pages)
print(len(chunks))

doc_embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document",
    google_api_key="AIzaSyBT_cXS1-V5ggaDcx7heSHJMb0h1r-xoPU",
)

db = Chroma.from_documents(
    chunks, doc_embeddings_model, persist_directory="./chroma_db"
)
