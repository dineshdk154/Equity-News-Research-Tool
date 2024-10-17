import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from dotenv import load_dotenv
import os



load_dotenv()  # Load environment variables from .env (especially OpenAI API key)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    if all(urls):  # Ensure all URL inputs are filled
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Check if data is loaded
        if not data:
            main_placeholder.text("No data loaded from URLs. Please check the URLs.")
        else:
            # Split data
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            # Check if documents are created
            if not docs:
                main_placeholder.text("No documents created after splitting. Please check the data.")
            else:
                st.write(f"Number of documents created: {len(docs)}")  # Check number of documents
                
                # Create embeddings
                embeddings = OpenAIEmbeddings()
                try:
                    embeddings_list = embeddings.embed_documents([doc.page_content for doc in docs])

                    # Check if embeddings are created
                    if not embeddings_list or len(embeddings_list) == 0:
                        main_placeholder.text("Failed to create embeddings from documents.")
                    else:
                        st.write(f"Number of embeddings created: {len(embeddings_list)}")  # Check number of embeddings

                        # Ensure embeddings list is non-empty
                        vectorstore_openai = FAISS.from_documents(docs, embeddings)
                        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                        time.sleep(2)

                        # Save the FAISS index to a pickle file
                        with open(file_path, "wb") as f:
                            pickle.dump(vectorstore_openai, f)

                except Exception as e:
                    main_placeholder.text(f"An error occurred while creating embeddings: {e}")
    
    else:
        main_placeholder.text("Please enter all three URLs.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            try:
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                st.header("Answer")
                st.write(result.get("answer", "No answer found."))

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)

            except Exception as e:
                st.error(f"An error occurred while retrieving the answer: {e}")
    else:
        st.warning("Please process URLs first.")
