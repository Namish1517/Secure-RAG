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

st.set_page_config(page_title="SecureRAG Banking", page_icon="🏦", layout="wide")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Setup Embeddings & DB
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ----- SIDEBAR -----
with st.sidebar:
    st.header("Admin Login")
    password = st.text_input("Enter Admin Password", type="password")
    
    if password == "admin123":
        st.session_state.authenticated = True
        st.success("Access Granted")
    elif password:
        st.error("Access Denied")
        st.session_state.authenticated = False

    st.divider()

    st.header("Security Settings")
    mode = st.radio("Select Mode:", ["Security Off", "Security On"], horizontal=True)

    st.divider()
    
    with st.expander("System Debug Info"):
        st.write("Authenticated:", st.session_state.authenticated)
        st.write("Mode:", mode)
        st.write("DB Loaded:", "db" in st.session_state)


# ----- MAIN PAGE -----
st.title("SecureRAG Chatbot")
st.markdown("Welcome to the SecureRAG Banking assistant")

# Document Upload Section
if st.session_state.authenticated:
    upload_msg_container = st.container()
    with st.expander("Manage Knowledge Base (Upload Documents)"):
        uploaded_file = st.file_uploader("Upload a document (.txt)", type=["txt"])

        if uploaded_file:
            with st.spinner("Processing document... Applying Layer 1 Security Check"):
                text = uploaded_file.read().decode("utf-8")

                if mode == "Security On":
                    is_safe, details = layer1_input_filter(text, uploaded_file.name)
                    if not is_safe:
                        upload_msg_container.error(f"Layer 1 Security Alert: Document rejected.\n{details}")
                        st.stop()
                    else:
                        upload_msg_container.success("Layer 1 Check Passed")

                # Save temporarily
                with open("temp.txt", "w", encoding="utf-8") as f:
                    f.write(text)

                # Load + split
                loader = TextLoader("temp.txt")
                documents = loader.load()

                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(documents)

                # Vector DB
                db = FAISS.from_documents(docs, embeddings)

                # Save DB in session
                st.session_state.db = db
                db.save_local("faiss_index")
                
                upload_msg_container.success(f"Document '{uploaded_file.name}' uploaded and processed into Vector Database.")

st.divider()

# ----- CHAT INTERFACE -----
# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if query := st.chat_input("Ask a banking question..."):
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if "db" not in st.session_state:
            error_msg = "Please upload documents first to establish a context."
            st.error(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        else:
            with st.status("Analyzing and Generating...", expanded=True) as status:
                st.write("🔍 Retrieving context from database...")
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

                pruned_warning = ""
                l3_warning = ""

                if mode == "Security On":
                    st.write("🛡️ Running Layer 2 Check (ZKIP Filtering)...")
                    is_safe, safe_chunks, details = layer2_retrieval_filter(query, retrieved_chunks, llm_generate_fn=zkip_groq_generator)
                    
                    if not is_safe:
                        status.update(label="Layer 2 Blocked!", state="error", expanded=True)
                        err_msg = f"**Layer 2 Security Alert:** Response blocked. {details}"
                        st.error(err_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
                        st.stop()
                    elif len(safe_chunks) < len(retrieved_chunks):
                        pruned_warning = f"⚠️ **Layer 2 Check:** Some context was pruned. {details}"
                        st.warning(pruned_warning)
                    else:
                        st.success("Layer 2 Check Passed")
                    
                    context = "\n".join(safe_chunks)
                else:
                    context = "\n".join(retrieved_chunks)

                st.write("🤖 Generating answer...")
                prompt = f"""
You are a banking assistant.

Answer ONLY using the context below.
Strictly adhere to the context provided below.
Context:
{context}

Question:
{query}
"""
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                raw_answer = response.choices[0].message.content

                if mode == "Security On":
                    st.write("🛡️ Running Layer 3 Check (Output Guard)...")
                    is_safe_out, final_answer, out_msg = layer3_output_filter(raw_answer)
                    
                    if not is_safe_out:
                        status.update(label="Layer 3 Blocked!", state="error", expanded=True)
                        st.error(f"Layer 3 Guard Block: {out_msg}")
                        blocked_msg = f"Response (BLOCKED): **{final_answer}**"
                        st.session_state.chat_history.append({"role": "assistant", "content": blocked_msg})
                        st.stop()
                    else:
                        if out_msg == "Apology Override":
                            l3_warning = "⚠️ **Layer 3 Interception:** Overwrote AI default apology loop."
                            st.warning(l3_warning)
                        else:
                            st.success("Layer 3 Check Passed")
                        actual_answer = final_answer
                else:
                    actual_answer = raw_answer
                
                status.update(label="Response Ready!", state="complete", expanded=False)
            
            # Surface warnings into the chat stream if they exist
            final_display = actual_answer
            if l3_warning:
                final_display = f"> {l3_warning}\n\n" + final_display
            if pruned_warning:
                final_display = f"> {pruned_warning}\n\n" + final_display
                
            st.markdown(final_display)
            st.session_state.chat_history.append({"role": "assistant", "content": final_display})
