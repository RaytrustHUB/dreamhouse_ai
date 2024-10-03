import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import requests
import replicate
import torch
from diffusers import StableDiffusionPipeline
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = Flask(__name__)

# Configuration for allowed extensions and uploads
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize async executor
executor = ThreadPoolExecutor()

# Ensure GPU is used if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to(device)

# API token for Replicate service
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Function to validate file extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Async function for generating image via Stable Diffusion
async def generate_stable_diffusion_image(prompt):
    try:
        result = pipe(prompt).images[0]
        return result
    except Exception as e:
        return {'error': f"Stable Diffusion generation failed: {str(e)}"}

# Async function for generating image via Replicate API
async def generate_replicate_image(prompt):
    try:
        output_url = replicate_client.models.get('stability-ai/stable-diffusion').predict(prompt=prompt)[0]
        return output_url
    except Exception as e:
        return {'error': f"Replicate API generation failed: {str(e)}"}

# Async function to enhance image using Real-ESRGAN via Replicate API
async def enhance_image_via_replicate(image_path):
    try:
        model = replicate_client.models.get('xinntao/realesrgan')
        output_url = model.predict(image=open(image_path, 'rb'))[0]
        return output_url
    except Exception as e:
        return {'error': f"Real-ESRGAN enhancement failed: {str(e)}"}

# Home route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image generation requests
@app.route('/generate', methods=['POST'])
async def generate_image():
    prompt = request.form.get('prompt')
    method = request.form.get('method', 'stable')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Generate image asynchronously using the appropriate method
    if method == 'replicate':
        result = await generate_replicate_image(prompt)
    else:
        result = await generate_stable_diffusion_image(prompt)

    if 'error' in result:
        return jsonify(result), 500

    # Save or serve the image
    image_url = result if isinstance(result, str) else url_for('static', filename='generated_image.png')
    return jsonify({'image_url': image_url}), 200

# Route to handle image enhancement requests
@app.route('/enhance', methods=['POST'])
async def enhance_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Enhance the image asynchronously
        result = await enhance_image_via_replicate(file_path)

        if 'error' in result:
            return jsonify(result), 500

        return jsonify({'enhanced_image_url': result}), 200

    return jsonify({'error': 'Invalid file type'}), 400

# Error handling
@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
