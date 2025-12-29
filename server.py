import os

from flask import Flask, request, jsonify
from agent import ingest_pdf, ask_question
from pyngrok import ngrok


app = Flask(__name__)

@app.route('/')
def home():
    return {"status": "API is running"}

@app.route('/ingest', methods=['POST'])
def ingest():
    data = request.json
    result = ingest_pdf(data['pdf_path'], data['project_type'])
    return jsonify(result)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    result = ask_question(data['question'], data['project_type'])
    return jsonify(result)

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))