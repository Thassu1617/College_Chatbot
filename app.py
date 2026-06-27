import os
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Automatically load configuration secrets from the local hidden .env file
load_dotenv()

app = Flask(__name__)

# 1. Initialize Local Vector Database Dependencies
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("college_db"):
    db = FAISS.load_local("college_db", embeddings, allow_dangerous_deserialization=True)
else:
    db = None
    print("WARNING: 'college_db' directory missing. Run create_db.py first.")

# 2. Setup Serverless Inference Client reading safely from Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")

model_id = "Qwen/Qwen2.5-7B-Instruct"  
print(f"Connecting to Serverless Cloud API for {model_id}...")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if db is None:
        return jsonify({"answer": "Database is offline. Please rebuild using create_db.py."})
        
    if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
        return jsonify({
            "answer": "⚠️ Configuration Error: The server cannot detect a valid HF_TOKEN environment variable."
        })
        
    try:
        user_query = request.json.get("question", "")
        if not user_query.strip():
            return jsonify({"answer": "Please ask a valid question."})

        # Fetch context blocks
        matched_docs = db.similarity_search(user_query, k=5)
        context_string = "\n\n".join([doc.page_content for doc in matched_docs])

        # Strict System Instructions
        system_instruction = (
            "You are 'College Dost', the official AI assistant for A1 Global Institute of Engineering and Technology.\n"
            "Strict Rules:\n"
            "1. Answer the user's question using ONLY the provided Context below.\n"
            "2. If the exact information is not explicitly present in the Context, say: 'I am sorry, I cannot find that specific detail in the college documents.'\n"
            "3. Do NOT make things up or use external knowledge.\n"
            "4. Provide costs strictly in Rupees. Never use dollars ($).\n"
            "5. CRITICAL: Clear separation of terms. If the user asks for 'courses' or 'branches' offered by the college, list ONLY the main degree programs (e.g., Computer Science and Engineering, ECE, Mechanical Engineering, AI&ML). Do NOT list individual semester subjects (like Cloud Computing, Automata, or Cryptography) as college courses."
            f"\n\nContext:\n{context_string}"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query}
        ]

        client = InferenceClient(model=model_id, token=HF_TOKEN)
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=250,
            temperature=0.3,
            frequency_penalty=1.0, 
        )
        
        clean_response = response.choices[0].message.content.strip()
        return jsonify({"answer": clean_response})

    except Exception as error_logs:
        return jsonify({"answer": f"An execution error occurred: {str(error_logs)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)