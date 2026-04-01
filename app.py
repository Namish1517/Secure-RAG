import streamlit as st
import os

from groq import Groq

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from layer1.filter import layer1_input_filter
from layer2.filter import layer2_retrieval_filter
from layer3.filter import layer3_output_filter

st.set_page_config(page_title="SecureRAG Banking", layout="wide")

st.title("SecureRAG Banking Chatbot")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if "db" not in st.session_state:
    try:
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.db = db
    except:
        pass
# -------------------------
# 🔑 GROQ SETUP
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# 🔐 AUTHENTICATION
# -------------------------
st.sidebar.header("Admin Login")

password = st.sidebar.text_input("Enter Admin Password", type="password")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if password == "admin123":
    st.session_state.authenticated = True
    st.sidebar.success("Access Granted")
elif password:
    st.sidebar.error("Access Denied")

# -------------------------
# ⚙️ SECURITY MODE
# -------------------------
st.header("Security Settings")

mode = st.radio(
    "Select Mode:",
    ["Security Off", "Security On"]
)

st.info(f"Current Mode: {mode}")

# -------------------------
# 📂 DOCUMENT UPLOAD
# -------------------------
st.header("Upload Documents")

if st.session_state.authenticated:
    uploaded_file = st.file_uploader("Upload a document", type=["txt"])

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

        if mode == "Security On":
            is_safe, details = layer1_input_filter(text, uploaded_file.name)
            if not is_safe:
                st.error(f"Layer 1 Security Alert: Document rejected.\n{details}")
                st.stop()
            else:
                st.success(f"Layer 1 Check Passed")

        # Save temporarily
        with open("temp.txt", "w", encoding="utf-8") as f:
            f.write(text)

        st.success("Document uploaded and processed")

        # Load + split
        loader = TextLoader("temp.txt")
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        # Embeddings (FREE)
        

        # Vector DB
        db = FAISS.from_documents(docs, embeddings)

        # Save DB in session
        st.session_state.db = db
        db.save_local("faiss_index")

else:
    st.warning("Admin access required to upload documents")

# -------------------------
# 💬 CHAT
# -------------------------
st.header("Chat with Banking Assistant")

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        if "db" not in st.session_state:
            st.error("Please upload documents first")
        else:
            retriever = st.session_state.db.as_retriever()
            docs = retriever.invoke(query)

            retrieved_chunks = [doc.page_content for doc in docs]
            
            # --- ZKIP GENERATOR CALLBACK ---
            def zkip_groq_generator(test_query, test_chunks):
                test_context = "\n".join(test_chunks)
                test_prompt = f"You are a banking assistant.\nAnswer ONLY using the context below.\nContext:\n{test_context}\nQuestion:\n{test_query}"
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": test_prompt}],
                    temperature=0.0
                )
                return resp.choices[0].message.content
            
            if mode == "Security On":
                is_safe, safe_chunks, details = layer2_retrieval_filter(query, retrieved_chunks, llm_generate_fn=zkip_groq_generator)
                if not is_safe:
                    st.error(f"Layer 2 Security Alert: Response blocked. {details}")
                    st.stop()
                elif len(safe_chunks) < len(retrieved_chunks):
                    st.warning(f"Layer 2 Check: Some context was pruned. {details}")
                else:
                    st.success("Layer 2 Check Passed")
                
                context = "\n".join(safe_chunks)
            else:
                context = "\n".join(retrieved_chunks)

            prompt = f"""
You are a banking assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

            # 🔥 GROQ CALL
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            raw_answer = response.choices[0].message.content

            if mode == "Security On":
                is_safe_out, final_answer, out_msg = layer3_output_filter(raw_answer)
                if not is_safe_out:
                    st.error(f"Layer 3 Guard Block: {out_msg}")
                    st.write("Response (BLOCKED):")
                    st.write(f"**{final_answer}**")
                else:
                    if out_msg == "Apology Override":
                        st.warning("Layer 3 Interception: Overrote AI default apology loop.")
                    else:
                        st.success("Layer 3 Check Passed")
                    st.write("Response:")
                    st.write(final_answer)
            else:
                st.write("Response:")
                st.write(raw_answer)
    else:
        st.warning("Enter a question")

# -------------------------
# 📊 DEBUG
# -------------------------
with st.expander("System Info"):
    st.write("Authenticated:", st.session_state.authenticated)
    st.write("Mode:", mode)
    st.write("DB Loaded:", "db" in st.session_state)