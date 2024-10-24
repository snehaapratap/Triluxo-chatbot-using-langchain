# **Chatbot using Langchain**

### **Project Overview**

This project showcases the development of a fully functional chatbot designed to extract data from web sources, embed the data into a vector store, and interact with users through a RESTful API. Built using the power of **Langchain**, **OpenAI**, and **Pinecone**, this chatbot efficiently handles user queries by retrieving the most relevant information from its vector store and offering concise, contextually relevant responses.

---

### **Key Features**

- **Data Extraction with Langchain**: Automatically scrapes and loads data from any web source via **Langchain's URL loaders**.
- **Embeddings with OpenAI**: Leverages **OpenAI’s embeddings** to represent the extracted data in high-dimensional vector space, optimizing for semantic search.
- **Vector Storage with Pinecone**: Efficiently stores embeddings in **Pinecone**, enabling fast and accurate retrieval of information based on user queries.
- **RESTful API with Flask**: Provides a user-friendly API interface for interacting with the chatbot, facilitating seamless integration with various applications.

---

### **Tech Stack**

- **Langchain**: Orchestrates the document loading, text splitting, and querying processes.
- **OpenAI**: Generates embeddings to represent textual data.
- **Pinecone**: Acts as a vector store to store and search embeddings.
- **Flask**: Provides a lightweight and scalable API framework.
- **Python**: The backbone of the entire application.

---

### **Installation Guide**

Follow these steps to set up and run the chatbot locally:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/snehaapratap/Triluxo-chatbot-using-langchain.git
   cd Triluxo-chatbot-using-langchain
   ```

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv langchain-chatbot-env
   source langchain-chatbot-env/bin/activate  # For Linux/Mac
   # OR
   langchain-chatbot-env\Scripts\activate     # For Windows
   ```

3. **Set Up Your API Keys**:

   - Get your **OpenAI API key** and **Pinecone API key** and set them as environment variables or directly in the scripts (replace `"YOUR_OPENAI_API_KEY"` and `"YOUR_PINECONE_API_KEY"` in `embedder.py`).
   
4. **Run the Embedding Creation Script**:

   ```bash
   python embedder.py
   ```

5. **Run the Flask API**:

   ```bash
   python app.py
   ```

6. **Test the API**:

   Send POST requests to `http://127.0.0.1:5000/chatbot` with the following format:

   ```json
   {
      "query": "What courses are available?"
   }
   ```

---

### **File Structure**

```
|-- langchain-chatbot
    |-- app.py             # Flask application for handling chatbot interaction
    |-- embedder.py        # Embedding generation and vector store integration
    |-- url_loader.py      # Data extraction from URL using Langchain
```

---

### **API Documentation**

**Endpoint**: `/chatbot`

- **Method**: `POST`
- **Request Body**:
  ```json
  {
     "query": "Your question here"
  }
  ```
- **Response**: A JSON response containing the chatbot's answer to the query.
  ```json
  {
     "response": "The answer to your query"
  }
  ```

---

### **Future Enhancements**

- **Scalability**: Expand the chatbot to extract data from multiple sources and domains.
- **Contextual Memory**: Enable the bot to retain the context of conversations for better multi-turn interactions.
- **UI Integration**: Integrate a front-end interface for a more user-friendly experience.

---

I hope you enjoy using the Chatbot and find it as exciting to work with as I did creating it!
