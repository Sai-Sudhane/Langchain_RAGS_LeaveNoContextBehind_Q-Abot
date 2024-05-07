import streamlit as st
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""Generate MCQ's with answer and solution for the aptitude topics given by user from RS Agarwal textbook for quantitative aptitude."""
        ),
        HumanMessagePromptTemplate.from_template(
            """Hi there! Could you please generate me 5 MCQ's with answers and solutions for the Topic. Topic:{topic} Answer:"""
        ),
    ]
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key="AIzaSyAEmi9I-lsNXfiFoUHq73cUKahSDwopAFo",
)

output_parser = StrOutputParser()

RAG_Chain = (
    {"topic": RunnablePassthrough()} | chat_template | chat_model | output_parser
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
    "<h1 class='title'>APTITUDE QUESTION GENERATOR💬</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='text'>Generate your MCQ's in seconds</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

prompt = st.text_input("Please mention the topic:")

if st.button("Generate"):
    response = RAG_Chain.invoke(prompt)
    st.write(response)
