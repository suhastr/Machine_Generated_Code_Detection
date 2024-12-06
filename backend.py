from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import traceback
import torch
from codeclassifier import CodeClassifier
import requests


# Model configuration constants

# below is the example, it's not finetuned, you have to replace this once you fine tune the model with appropriate dataset

# PERFORMER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
# OBSERVER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M"

# Initialize Flask app
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Determine the computing device (GPU if available, otherwise CPU)
computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the Hugging Face authentication token
with open("hugging_face_auth_token.txt") as auth_token_file:
    hugging_face_auth_token = auth_token_file.readline().strip()

# Initialize the detectors
code_analysis_detector = CodeClassifier(
    OBSERVER_MODEL_NAME, 
    PERFORMER_MODEL_NAME, 
    hugging_face_auth_token
)

# Backend API endpoint for proxying requests
BACKEND_API_ENDPOINT = "https://ondemand.orc.gmu.edu/rnode/gpu027.orc.gmu.edu/27304/proxy/5000/"

# Define the home route
@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def proxy_analysis_request():
    """
    Proxy route to forward analysis requests to the backend API.
    """
    request_headers = {
        'Content-Type': 'application/json',
    }
    request_data = request.json

    try:
        # Forward the request to the backend API and return its response
        backend_response = requests.post(BACKEND_API_ENDPOINT, headers=request_headers, json=request_data)
        return jsonify(backend_response.json()), backend_response.status_code
    except Exception as exception:
        logger.error(f"Error proxying request: {str(exception)}")
        return jsonify({'error': 'Failed to connect to backend API'}), 500

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_content():
    """
    Main route for analyzing text or code content.
    """
    if request.method == 'OPTIONS':
        # Handle preflight OPTIONS request
        return '', 204

    try:
        request_data = request.json
        if not request_data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400

        content_type = request_data.get('type')
        content_payload = request_data.get('content')

        # Handle analysis based on content type
        if content_type == "code":
            return analyze_code_content(content_payload)
        else:
            logger.error(f"Invalid content type: {content_type}")
            return jsonify({'error': 'Invalid content type'}), 400
    except Exception as exception:
        logger.error(f"Unexpected error: {str(exception)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(exception)}'}), 500

def analyze_code_content(code_data):
    """
    Analyze code content using the CodeClassifier model.
    """
    try:
        # classify_code using the code detector
        ai_generated_status, evaluation_score = code_analysis_detector.classify_code(code_data, computation_device)
        
        # Prepare the response
        analysis_result = {
            'CodeClassifier': {
                'is_ai_generated': ai_generated_status,
                'score': float(evaluation_score),
                'result': f"{ai_generated_status} (Score: {float(evaluation_score):.4f})"
            }
        }
        return jsonify(analysis_result)
    except Exception as exception:
        logger.error(f"Error analyzing code: {str(exception)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to analyze code'}), 500

# Entry point for running the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
