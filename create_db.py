import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_vector_db():
    if not os.path.exists("college_info.txt"):
        print("Error: 'college_info.txt' not found!")
        return

    print("Step 1: Reading college text file...")
    with open("college_info.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    print("Step 2: Fragmenting text into clean informational blocks...")
    # Smaller chunk size ensures short definitions at the bottom are preserved perfectly
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    print("Step 3: Initializing embedding transformer model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Step 4: Transforming text blocks into vector index arrays...")
    db = FAISS.from_texts(texts, embeddings)

    print("Step 5: Exporting vector index files into 'college_db' folder...")
    db.save_local("college_db")
    print("\nInitialization Complete! The optimized 'college_db' is ready.")

if __name__ == "__main__":
    create_vector_db()