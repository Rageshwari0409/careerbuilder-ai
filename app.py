from flask import Flask
from routes.routes import register_routes
from utils.logger import setup_logging
# from config.config import UPLOAD_FOLDER
import logging
import os
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

setup_logging()  # Set up logging
register_routes(app)

if __name__ == "__main__":
    app.run(debug=True)