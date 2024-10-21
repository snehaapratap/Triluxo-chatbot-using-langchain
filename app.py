from flask import Flask, request, jsonify
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from embedder import create_embeddings_and_store

app = Flask(__name__)

llm = OpenAI()
index_name = "langchain-chatbot-index"

@app.route("/chatbot", methods=["POST"])
def chatbot():
    query = request.json.get("query", "")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    chain = load_qa_chain(llm, vectorstore=index_name)
    response = chain.run(query)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
