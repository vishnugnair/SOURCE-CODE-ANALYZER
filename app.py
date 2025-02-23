from flask import Flask, render_template, request
from dotenv import load_dotenv
from src.helper import clone_repo, load_repo, text_splitter, load_embedding
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
import os

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

app = Flask(__name__)

# Global Variables (to store the QA chain and memory)
qa = None
memory = None

@app.route("/", methods=["GET", "POST"])
def index():
    global qa, memory  # Use global variables so they persist across requests

    if request.method == "POST":
        repo_url = request.form["repo_url"]  # Get repo URL from user input

        if not repo_url:
            return render_template("index.html", error="⚠️ Please enter a valid GitHub repo URL.")

        repo_path = "repo/"  # Set repo storage path
        clone_repo(repo_url, repo_path)  # Clone the repo when the user submits it

        # Load and process repository files
        documents = load_repo(repo_path)
        text_chunks = text_splitter(documents)
        embeddings = load_embedding()

        # Store embeddings in ChromaDB
        vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory="./db")
        vectordb.persist()

        # Initialize LLM and conversation memory
        llm = ChatOpenAI()
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

        # Create retrieval-based chat chain
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
            return_source_documents=True
        )

        return render_template("index.html", success="✅ Repository cloned and processed successfully!", repo_url=repo_url)

    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global qa, memory  # Ensure we access the stored QA chain

    if qa is None:
        return render_template("index.html", error="⚠️ Please clone a GitHub repo first before asking questions.")

    question = request.form["question"]
    result = qa({"question": question, "chat_history": memory.chat_memory.messages})
    response = result["answer"]
    
    return render_template("index.html", query=question, response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
