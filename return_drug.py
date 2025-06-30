from __future__ import annotations
import base64
import os
from typing import List
from flask import Flask, request, jsonify
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import datetime
import shutil
from keys import GOOGLE_API_KEY
import tempfile
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
globals.set_debug(False)

parser = None

# Model for medicine name only
class MedicineName(BaseModel):
    name: str

# Image loader for URLs
def load_images_from_urls(inputs: dict) -> dict:
    image_urls = inputs["image_urls"]
    def encode_image_from_url(image_url):
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to download image from URL: {str(e)}")

    images_base64 = [encode_image_from_url(url) for url in image_urls]
    return {"images": images_base64}

# Image loader for local files (keeping for compatibility)
def load_images(inputs: dict) -> dict:
    image_paths = inputs["image_paths"]
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    images_base64 = [encode_image(image_path) for image_path in image_paths]
    return {"images": images_base64}

load_images_chain = TransformChain(
    input_variables=["image_paths"],
    output_variables=["images"],
    transform=load_images
)

load_images_from_urls_chain = TransformChain(
    input_variables=["image_urls"],
    output_variables=["images"],
    transform=load_images_from_urls
)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        model_kwargs={"generation_config": {"temperature": 0.3, "max_output_tokens": 1024}}
    )

    content = [
        {"type": "text", "text": inputs['prompt']},
        {"type": "text", "text": parser.get_format_instructions()}
    ]
    for img in inputs['images']:
        content.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{img}"
        })
    msg = model.invoke([HumanMessage(content=content)])
    return msg.content

def get_medicine_name_from_url(image_url: str) -> dict:
    global parser
    parser = JsonOutputParser(pydantic_object=MedicineName)

    prompt = """
    You are given an image of a medicine box. Your task is to extract the exact name of the medicine as it appears on the package.
    Do not infer or add anything. Only return the medicine name, and leave empty if not clear.
    """

    chain = load_images_from_urls_chain | image_model | parser
    return chain.invoke({'image_urls': [image_url], 'prompt': prompt})

def get_medicine_name(image_paths: List[str]) -> dict:
    global parser
    parser = JsonOutputParser(pydantic_object=MedicineName)

    prompt = """
    You are given an image of a medicine box. Your task is to extract the exact name of the medicine as it appears on the package.
    Do not infer or add anything. Only return the medicine name, and leave empty if not clear.
    """

    chain = load_images_chain | image_model | parser
    return chain.invoke({'image_paths': image_paths, 'prompt': prompt})

def remove_temp_folder(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

# API Routes
@app.route('/', methods=['GET'])
def info():
    """API information endpoint"""
    return jsonify({
        "name": "Medical Prescription Parser API",
        "version": "1.0.0",
        "description": "API for extracting medicine names from images using AI",
        "endpoints": {
            "/": "Get API information",
            "/analyze": "Analyze medicine image (POST with image URL)"
        },
        "usage": {
            "/analyze": {
                "method": "POST",
                "content_type": "application/json",
                "parameters": {
                    "image_url": "URL of the image (png, jpg, jpeg)"
                },
                "example": {
                    "image_url": "https://example.com/medicine.jpg"
                },
                "response": {
                    "success": "boolean",
                    "medicine_name": "string",
                    "message": "string"
                }
            }
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_medicine():
    """Main endpoint for analyzing medicine images from URL"""
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data provided",
                "medicine_name": None
            }), 400

        # Check if image URL is provided
        if 'image_url' not in data:
            return jsonify({
                "success": False,
                "message": "No image_url provided in JSON data",
                "medicine_name": None
            }), 400

        image_url = data['image_url']

        if not image_url or not isinstance(image_url, str):
            return jsonify({
                "success": False,
                "message": "Invalid image_url provided",
                "medicine_name": None
            }), 400

        # Validate URL format
        if not (image_url.startswith('http://') or image_url.startswith('https://')):
            return jsonify({
                "success": False,
                "message": "Image URL must start with http:// or https://",
                "medicine_name": None
            }), 400

        # Check if URL points to an image
        allowed_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        if not any(image_url.lower().endswith(ext) for ext in allowed_extensions):
            # Try to validate by checking content-type
            try:
                head_response = requests.head(image_url, timeout=10)
                content_type = head_response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    return jsonify({
                        "success": False,
                        "message": "URL does not point to a valid image file",
                        "medicine_name": None
                    }), 400
            except:
                pass  # If head request fails, we'll try anyway

        # Analyze the image
        result = get_medicine_name_from_url(image_url)

        # Return result
        medicine_name = result.get("name", "")

        if medicine_name and medicine_name.strip():
            return jsonify({
                "success": True,
                "message": "Medicine name extracted successfully",
                "medicine_name": medicine_name.strip()
            })
        else:
            return jsonify({
                "success": False,
                "message": "Could not extract medicine name from the image",
                "medicine_name": None
            })

    except requests.exceptions.RequestException as e:
        return jsonify({
            "success": False,
            "message": f"Error downloading image: {str(e)}",
            "medicine_name": None
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}",
            "medicine_name": None
        }), 500

@app.route('/info', methods=['GET'])
def home():
    """Alternative info endpoint"""
    return jsonify({
        "message": "Welcome to Medical Prescription Parser API",
        "info": "Visit / for API documentation"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
