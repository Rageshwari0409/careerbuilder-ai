from flask import request, jsonify, render_template
from services.chat_service import handle_chat
from services.rag_service import handle_rag_flow
import os
import uuid
from werkzeug.utils import secure_filename
from utils.logger import setup_logging
import logging

setup_logging()

def register_routes(app):
    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/upload_pdf", methods=["POST"])
    def upload_pdf():
        logging.info("Received file upload request.")
        file = request.files.get("file")
        logging.info(f"Uploaded file: {file.filename if file else 'No file'}")
        if not file:
            return jsonify({"message": "No file uploaded"}), 400
        filename = secure_filename(file.filename)
        logging.info(f"Secure filename: {filename}")
        unique_name = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        try:
            file.save(file_path)
            logging.info(f"File saved to {file_path}")
            return jsonify({"message": f"File {filename} uploaded successfully", "file_path": file_path})
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            return jsonify({"message": "Error saving file"}), 500

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.json
        return handle_chat(data)