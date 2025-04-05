import streamlit as st
import os
import re
import json
import hashlib
import tempfile
import shutil
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv
import tiktoken
import numpy as np
from typing import Dict, Any, List
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Create directories if they don't exist
VECTOR_STORE_DIR = "vector_stores"
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)


def get_document_hash(pdf_name):
    """Generate a unique hash for the document name."""
    return hashlib.md5(pdf_name.encode("utf-8")).hexdigest()


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_pdf_text(pdf_file):
    """Extract text from PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    os.unlink(tmp_file_path)
    return pages


def get_text_chunks(pages):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    return chunks


def get_vectorstore(text_chunks, pdf_name):
    """Create and save vectorstore for the document."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    # Create document-specific directory with hash name
    doc_hash = get_document_hash(pdf_name)
    doc_dir = os.path.join(VECTOR_STORE_DIR, doc_hash)
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)

    # Use a temporary directory with ASCII path
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save to temporary directory
        vectorstore.save_local(temp_dir)

        # Copy all files to the final location
        for file in os.listdir(temp_dir):
            src = os.path.join(temp_dir, file)
            dst = os.path.join(doc_dir, file)
            shutil.copy2(src, dst)

    # Save original filename
    with open(os.path.join(doc_dir, "original_name.txt"), "w", encoding="utf-8") as f:
        f.write(pdf_name)

    return vectorstore


def load_vectorstore(pdf_name):
    """Load vectorstore for the document."""
    doc_hash = get_document_hash(pdf_name)
    doc_dir = os.path.join(VECTOR_STORE_DIR, doc_hash)

    if os.path.exists(doc_dir) and os.path.exists(os.path.join(doc_dir, "index.faiss")):
        # Use a temporary directory with ASCII path
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy all files to temporary directory
            for file in os.listdir(doc_dir):
                if file != "original_name.txt":  # Skip the original name file
                    src = os.path.join(doc_dir, file)
                    dst = os.path.join(temp_dir, file)
                    shutil.copy2(src, dst)

            # Load from temporary directory
            embeddings = OpenAIEmbeddings()
            return FAISS.load_local(temp_dir, embeddings)
    return None


def get_conversation_chain(vectorstore):
    """Create conversation chain for the vectorstore."""
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Custom retriever to get source documents
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Get top 3 most relevant chunks
    )

    # Create a custom chain that handles source documents
    class CustomConversationChain(ConversationalRetrievalChain):
        def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            result = super()._call(inputs)
            # Store source documents in a separate session state
            if "source_documents" in result:
                st.session_state.last_source_documents = result["source_documents"]
            # Return only the answer
            return {"answer": result["answer"]}

    conversation_chain = CustomConversationChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )
    return conversation_chain


def get_combined_vectorstore(vectorstores):
    """Combine multiple vectorstores into one."""
    if not vectorstores:
        return None

    # Get the first vectorstore
    combined_store = list(vectorstores.values())[0]

    # Merge with other vectorstores
    for vectorstore in list(vectorstores.values())[1:]:
        combined_store.merge_from(vectorstore)

    return combined_store


def get_saved_documents():
    """Get list of saved documents from vector store directory."""
    if not os.path.exists(VECTOR_STORE_DIR):
        return []

    documents = []
    for item in os.listdir(VECTOR_STORE_DIR):
        if os.path.isdir(os.path.join(VECTOR_STORE_DIR, item)):
            # Check if it's a valid vector store directory
            if os.path.exists(os.path.join(VECTOR_STORE_DIR, item, "index.faiss")):
                # Try to get original name
                original_name_file = os.path.join(
                    VECTOR_STORE_DIR, item, "original_name.txt"
                )
                if os.path.exists(original_name_file):
                    with open(original_name_file, "r", encoding="utf-8") as f:
                        original_name = f.read().strip()
                        documents.append((item, original_name))
                else:
                    documents.append((item, item))
    return documents


def main():
    st.set_page_config(page_title="PDF Q&A Service", page_icon="📚")
    st.header("PDF Q&A Service 📚")

    # Initialize session state
    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = {}
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = set()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "last_source_documents" not in st.session_state:
        st.session_state.last_source_documents = None
    if "saved_docs" not in st.session_state:
        st.session_state.saved_docs = get_saved_documents()
        print(f"Initialized with {len(st.session_state.saved_docs)} saved documents")
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = []

    # Load existing vectorstores
    for doc_hash, original_name in st.session_state.saved_docs:
        if doc_hash not in st.session_state.vectorstores:
            vectorstore = load_vectorstore(original_name)
            if vectorstore:
                st.session_state.vectorstores[doc_hash] = vectorstore
                print(f"Loaded vectorstore: {original_name}")

    # Sidebar for document management
    with st.sidebar:
        st.subheader("📄 Document Management")

        # File upload
        pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

        if pdf_file is not None:
            pdf_name = pdf_file.name.replace(".pdf", "")
            doc_hash = get_document_hash(pdf_name)

            # Check if vector store already exists
            existing_vectorstore = load_vectorstore(pdf_name)
            if existing_vectorstore:
                st.info("Using existing vector store for this PDF.")
                st.session_state.vectorstores[doc_hash] = existing_vectorstore
                st.session_state.selected_docs.add(doc_hash)
                # Create combined vectorstore with selected documents
                combined_store = get_combined_vectorstore(
                    {
                        k: v
                        for k, v in st.session_state.vectorstores.items()
                        if k in st.session_state.selected_docs
                    }
                )
                st.session_state.conversation = get_conversation_chain(combined_store)
                debug_msg = f"Added existing document: {pdf_name}"
                print(debug_msg)
                st.session_state.debug_info.append(debug_msg)
            else:
                with st.spinner("Processing PDF..."):
                    # Get PDF text
                    pages = get_pdf_text(pdf_file)

                    # Calculate total tokens
                    total_text = " ".join([page.page_content for page in pages])
                    total_tokens = num_tokens_from_string(total_text)

                    # Get text chunks
                    text_chunks = get_text_chunks(pages)

                    # Calculate tokens per chunk
                    chunk_tokens = [
                        num_tokens_from_string(chunk.page_content)
                        for chunk in text_chunks
                    ]

                    # Display token information
                    st.info(f"Total tokens in PDF: {total_tokens:,}")
                    st.info(f"Number of chunks: {len(text_chunks)}")
                    st.info(
                        f"Average tokens per chunk: {sum(chunk_tokens)/len(chunk_tokens):.1f}"
                    )
                    st.info(f"Min tokens in chunk: {min(chunk_tokens)}")
                    st.info(f"Max tokens in chunk: {max(chunk_tokens)}")

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks, pdf_name)
                    st.session_state.vectorstores[doc_hash] = vectorstore
                    st.session_state.selected_docs.add(doc_hash)
                    # Create combined vectorstore with selected documents
                    combined_store = get_combined_vectorstore(
                        {
                            k: v
                            for k, v in st.session_state.vectorstores.items()
                            if k in st.session_state.selected_docs
                        }
                    )
                    st.session_state.conversation = get_conversation_chain(
                        combined_store
                    )
                    debug_msg = f"Added new document: {pdf_name}"
                    print(debug_msg)
                    st.session_state.debug_info.append(debug_msg)
                    st.success("PDF processed and vector store saved successfully!")

        # Show saved documents with checkboxes
        if st.session_state.saved_docs:
            st.subheader("📚 Select Documents for Q&A")
            for doc_hash, original_name in st.session_state.saved_docs:
                if doc_hash in st.session_state.vectorstores:
                    # Create a checkbox for each document
                    is_selected = st.checkbox(
                        f"📄 {original_name}",
                        value=doc_hash in st.session_state.selected_docs,
                        key=f"select_{doc_hash}",
                    )

                    # Update selected documents based on checkbox state
                    if is_selected and doc_hash not in st.session_state.selected_docs:
                        st.session_state.selected_docs.add(doc_hash)
                        debug_msg = f"Selected document: {original_name}"
                        print(debug_msg)
                        st.session_state.debug_info.append(debug_msg)
                        # Update conversation chain with new selection
                        combined_store = get_combined_vectorstore(
                            {
                                k: v
                                for k, v in st.session_state.vectorstores.items()
                                if k in st.session_state.selected_docs
                            }
                        )
                        st.session_state.conversation = get_conversation_chain(
                            combined_store
                        )
                    elif not is_selected and doc_hash in st.session_state.selected_docs:
                        st.session_state.selected_docs.remove(doc_hash)
                        debug_msg = f"Deselected document: {original_name}"
                        print(debug_msg)
                        st.session_state.debug_info.append(debug_msg)
                        # Update conversation chain with new selection
                        if st.session_state.selected_docs:
                            combined_store = get_combined_vectorstore(
                                {
                                    k: v
                                    for k, v in st.session_state.vectorstores.items()
                                    if k in st.session_state.selected_docs
                                }
                            )
                            st.session_state.conversation = get_conversation_chain(
                                combined_store
                            )
                        else:
                            st.session_state.conversation = None

        # Display selected documents
        if st.session_state.selected_docs:
            st.subheader("Selected Documents")
            for doc_hash in st.session_state.selected_docs:
                original_name = next(
                    (
                        original
                        for hash, original in st.session_state.saved_docs
                        if hash == doc_hash
                    ),
                    doc_hash,
                )
                st.write(f"📄 {original_name}")
        else:
            st.write("❌ No documents selected")
            st.write("Please select documents from the list above")

    # Main content area
    st.subheader("💬 Chat with your PDF")

    # Debug information section
    with st.expander("🔍 Debug Information"):
        st.write("### Session State")
        st.write("#### Selected Documents")
        if st.session_state.selected_docs:
            for doc_hash in st.session_state.selected_docs:
                original_name = next(
                    (
                        original
                        for hash, original in st.session_state.saved_docs
                        if hash == doc_hash
                    ),
                    doc_hash,
                )
                st.write(f"- {original_name} (Hash: {doc_hash})")
        else:
            st.write("No documents selected")

        st.write("#### Available Vectorstores")
        if st.session_state.vectorstores:
            for doc_hash in st.session_state.vectorstores:
                original_name = next(
                    (
                        original
                        for hash, original in st.session_state.saved_docs
                        if hash == doc_hash
                    ),
                    doc_hash,
                )
                st.write(f"- {original_name} (Hash: {doc_hash})")
        else:
            st.write("No vectorstores available")

        st.write("#### Conversation State")
        st.write(
            f"- Conversation initialized: {st.session_state.conversation is not None}"
        )
        st.write(f"- Current question: {st.session_state.current_question or 'None'}")
        st.write(f"- Processing: {st.session_state.processing}")

        if st.session_state.debug_info:
            st.write("#### Recent Debug Logs")
            for log in st.session_state.debug_info[-5:]:  # Show last 5 logs
                st.write(f"- {log}")

    # Create a container for the question input
    with st.container():
        st.write("---")  # Separator line
        # Create the text input
        user_question = st.text_input(
            "Ask a question about your PDF:", key="question_input"
        )

    # Handle question submission
    if user_question and not st.session_state.processing:
        st.session_state.current_question = user_question
        debug_msg = f"Received question: {user_question}"
        print(debug_msg)
        st.session_state.debug_info.append(debug_msg)

        if not st.session_state.selected_docs:
            st.warning("Please select at least one document first.")
            st.session_state.debug_info.append(
                f"Warning: No documents selected for question: {user_question}"
            )
        elif st.session_state.conversation is None:
            # Initialize conversation chain if not exists
            debug_msg = f"Initializing conversation chain with {len(st.session_state.selected_docs)} documents"
            print(debug_msg)
            st.session_state.debug_info.append(debug_msg)

            combined_store = get_combined_vectorstore(
                {
                    k: v
                    for k, v in st.session_state.vectorstores.items()
                    if k in st.session_state.selected_docs
                }
            )
            st.session_state.conversation = get_conversation_chain(combined_store)
        else:
            debug_msg = f"Processing question for {len(st.session_state.selected_docs)} selected documents"
            print(debug_msg)
            st.session_state.debug_info.append(debug_msg)

            st.session_state.processing = True
            with st.spinner("Thinking..."):
                # Create a container for the streaming response
                response_container = st.container()
                with response_container:
                    # Initialize streaming response
                    response_placeholder = st.empty()
                    full_response = ""

                    try:
                        # Get streaming response
                        for chunk in st.session_state.conversation.stream(
                            {"question": user_question}
                        ):
                            if "answer" in chunk:
                                full_response += chunk["answer"]
                                response_placeholder.write(
                                    f"Assistant: {full_response}"
                                )

                        debug_msg = (
                            f"Generated response for question: {user_question[:50]}..."
                        )
                        print(debug_msg)
                        st.session_state.debug_info.append(debug_msg)

                        # Display source documents if available
                        if st.session_state.last_source_documents:
                            st.write("---")
                            st.write("📚 **Related Documents:**")
                            for j, doc in enumerate(
                                st.session_state.last_source_documents, 1
                            ):
                                similarity = doc.metadata.get("score", "N/A")
                                if isinstance(similarity, (int, float)):
                                    similarity_str = f"{similarity:.2f}"
                                else:
                                    similarity_str = str(similarity)
                                st.write(
                                    f"**Source {j}** (Similarity: {similarity_str})"
                                )
                                st.write(doc.page_content)
                                st.write("---")
                                debug_msg = f"Source document {j} from {doc.metadata.get('source', 'unknown')} (Similarity: {similarity_str})"
                                st.session_state.debug_info.append(debug_msg)
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        print(error_msg)
                        st.session_state.debug_info.append(error_msg)
                    finally:
                        st.session_state.processing = False
                        # Clear the current question
                        st.session_state.current_question = None


if __name__ == "__main__":
    main()
