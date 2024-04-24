import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

key = st.secrets["GEMINI_API_KEY"]
doc_embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document",
    google_api_key=key,
)

db_connection = Chroma(
    persist_directory="./chroma_db", embedding_function=doc_embeddings_model
)

retriever = db_connection.as_retriever(search_kwargs={"k": 110})

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an useful AI bot, answer any question asked by the user from the specific information regarding it. Do answer anything apart from that"""
        ),
        HumanMessagePromptTemplate.from_template(
            """Answer the following question based on the specific context. Context:{context} Question:{question} Answer:"""
        ),
    ]
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=key,
)

output_parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


RAG_Chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

custom_css = """
<style>
.title {
    color:#ffa500 ;
}

.text {
    color: #00ff00;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(
    "<h1 class='title'>LEAVE NO CONTEXT BEHIND - Q&A ASSISTANT💬</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='text'>Get your questions regarding the \"Leave No Context Behind\" paper answered in seconds</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

prompt = st.text_input("Ask you question about the research paper..")

if st.button("Ask"):
    response = RAG_Chain.invoke(prompt)
    st.write(response)
