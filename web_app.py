import os
import sys
from flask import Flask, send_from_directory

# Initialize Flask app configured to serve static assets from the root directory
app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    print("Starting local development asset server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
