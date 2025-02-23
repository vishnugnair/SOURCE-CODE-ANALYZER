import os
from git import Repo
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def clone_repo(url, repo_path="repo/"):
    """Clones a GitHub repository when the user submits a URL."""
    if os.path.exists(repo_path):
        print("✅ Repository already exists. Skipping cloning.")
        return
    os.makedirs(repo_path, exist_ok=True)
    Repo.clone_from(url, to_path=repo_path)
    print("✅ Repository cloned successfully!")

def load_repo(repo_path):
    """Loads code files from the cloned repo."""
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],  # Change to ".js" for JavaScript
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} documents from {repo_path}")
    return documents

def text_splitter(documents):
    """Splits documents into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = splitter.split_documents(documents)
    print(f"🔹 Created {len(text_chunks)} text chunks")
    return text_chunks

def load_embedding():
    """Loads OpenAI embeddings."""
    return OpenAIEmbeddings(disallowed_special=())
