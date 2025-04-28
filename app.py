import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain


# Streamlit app setup
st.title("Extract Website info and ask question about it from AI!")


url = st.text_input("Enter your URL:")

if url:
    # Progress indicator
    with st.spinner("Setting up models and chains..."):
        try:
            groq_api_key = "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"
            # Create a Groq API client
            llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
            # Define embeddings model
            model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Load data
            loader = WebBaseLoader(url)
            data = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)


            # Create FAISS vector store
            db = FAISS.from_documents(chunks, model)

            # Initialize LLM with Groq API
            llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

            # Create retriever
            retriever = db.as_retriever()

            # Define prompt
            prompt = ChatPromptTemplate.from_template(
                """
                Answer the following question based only on the provided context:
                <context>
                {context}
                </context>
                """
            )

            chain = create_stuff_documents_chain(llm,prompt)

            # Create chain
            retriever_chain = create_retrieval_chain(retriever,chain)

            # Display result
            st.success("Information Retrieved!")

            query = st.text_input("Ask question from AI about the website..")

            if query:
                with st.spinner("Retrieving from AI..."):
                    response = retriever_chain.invoke({"input":query})

                    st.success("Answer retrived from AI !")
                    st.write(response['answer'])
        

        except Exception as e:
            st.error(f"An error occurred: {e}")
