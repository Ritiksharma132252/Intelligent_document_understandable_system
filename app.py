# app.py
from flask import Flask, render_template, request, jsonify
from llm_service import create_vector_store, generate_rag_response, FAISS_INDEX_PATH
import os


app = Flask(__name__)
# Set a temporary directory for uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Renders the main HTML page and handles file upload."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'pdf_file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files['pdf_file']
        
        # If the user submits an empty part
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # Check file extension and save it
        if file and file.filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:
                # 1. Create the vector store from the uploaded PDF
                num_chunks = create_vector_store(file_path)
                
                # 2. Remove the temp file after processing
                os.remove(file_path)
                
                return jsonify({
                    "status": "success", 
                    "message": f"Document processed! {num_chunks} chunks created. You can now ask questions."
                }), 200
            except Exception as e:
                return jsonify({"status": "error", "message": f"Processing failed: {e}"}), 500
        
        return jsonify({"status": "error", "message": "Invalid file format. Please upload a PDF."}), 400

    # Render the main page for GET requests
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query_document():
    """Handles the user's question and returns the RAG response."""
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty."}), 400
    
    # 1. Check if the vector store is ready
    if not os.path.exists(FAISS_INDEX_PATH):
        return jsonify({"status": "error", "message": "No document loaded. Please upload and process a PDF first."}), 400

    # 2. Generate the AI response using RAG
    ai_response = generate_rag_response(query)
    
    return jsonify({
        "status": "success",
        "response": ai_response
    }), 200


if __name__ == '__main__':
    # Cleanup any old index file on startup for a clean run
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    
    # Run the Flask app
    # Use 0.0.0.0 for potential Docker deployment if needed later
    app.run(debug=True, host='0.0.0.0')