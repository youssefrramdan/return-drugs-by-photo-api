

from __future__ import annotations
import base64
import os
from typing import List
from flask import Flask, request, render_template
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

app = Flask(__name__)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
globals.set_debug(False)

parser = None

# ✅ Model for medicine name only
class MedicineName(BaseModel):
    name: str

# ✅ Image loader
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

# ✅ Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if uploaded_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = uploaded_file.filename.split('.')[0].replace(' ', '_')
            output_folder = os.path.join(".", f"Check_{filename}_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)

            file_path = os.path.join(output_folder, uploaded_file.filename)
            uploaded_file.save(file_path)

            result = get_medicine_name([file_path])

            remove_temp_folder(output_folder)

            return render_template('result.html', medicine_name=result.get("name", "Not found"))

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
