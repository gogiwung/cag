import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


def get_pdf_text(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    os.unlink(tmp_file_path)
    return pages


def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def main():
    st.set_page_config(page_title="PDF Q&A Service", page_icon="📚")
    st.header("PDF Q&A Service 📚")

    # File upload
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if pdf_file is not None:
        with st.spinner("Processing PDF..."):
            # Get PDF text
            pages = get_pdf_text(pdf_file)

            # Get text chunks
            text_chunks = get_text_chunks(pages)

            # Create vector store
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.vectorstore = vectorstore

            # Create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("PDF processed successfully!")

    # Chat interface
    if st.session_state.conversation is not None:
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": user_question})
                st.session_state.chat_history = response["chat_history"]

                # Display chat history
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(f"Human: {message.content}")
                    else:
                        st.write(f"Assistant: {message.content}")


if __name__ == "__main__":
    main()
