import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
from flask_cors import CORS
import requests

app = Flask(__name__)
# CORS configuration
cors_options = {
    "origins": "https://www.healthlens.app",  # Allow requests from your frontend
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Allow these methods
    "allow_headers": ["Content-Type", "Authorization"],  # Allow these headers
    "supports_credentials": True  # If you need to support credentials
}

CORS(app, **cors_options)  # Apply CORS to the app

# Function to download model from Google Drive
def download_model_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        if 'download_warning' in response.cookies:
            return response.cookies['download_warning']
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768  # 32KB
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Model file IDs from Google Drive
MODEL_FILES = {
    'melanoma': {
        'file_id': '1Mo6c60Vyf5QicHKa9eO9F2piIOZNgG44',
        'model': None
    },
    'dermnet': {
        'file_id': '1bHxkOHMKex1kGR9lCAjwhwCsWXLsg6sW',
        'model': None
    },
    'nail': {
        'file_id': '1G7wSTJdb2s46xcukTS_NZzN5TD6VhF9L',
        'model': None
    },
    'cataract': {
        'file_id': '1WYRzO4IwAvxwn3cV_8nv3QDhT6NAV8Z1',
        'model': None
    }
}

# Load the models once on startup
@app.before_request
def load_models():
    device = torch.device('cpu')
    for name, info in MODEL_FILES.items():
        model_path = f"{name}_classifier_10_epochs.pth"
        if not os.path.exists(model_path):
            print(f"Downloading {name} model...")
            download_model_from_google_drive(info['file_id'], model_path)
            print(f"{name} model downloaded.")
        else:
            print(f"{name} model already exists.")

        # Load the model architecture
        model = models.mobilenet_v2(weights='DEFAULT')
        if name == 'melanoma':
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.6, inplace=False),
                nn.Linear(in_features=1280, out_features=2, bias=True),
                nn.LogSoftmax(dim=1)
            )
        elif name == 'dermnet':
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.6, inplace=False),
                nn.Linear(in_features=1280, out_features=23, bias=True),
                nn.LogSoftmax(dim=1)
            )
        elif name == 'nail':
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.6, inplace=False),
                nn.Linear(in_features=1280, out_features=17, bias=True),
                nn.LogSoftmax(dim=1)
            )
        elif name == 'cataract':
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.6, inplace=False),
                nn.Linear(in_features=1280, out_features=2, bias=True),
                nn.LogSoftmax(dim=1)
            )

        # Load the state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Save the model in the dictionary
        MODEL_FILES[name]['model'] = model

    print("All models loaded successfully.")

# Preprocessing function
def preprocess_image(file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Expected input size
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  
    ])
    image = Image.open(file).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Endpoint to upload an image and get predictions
@app.route('/predict/skin', methods=['OPTIONS', 'POST'])
def predict_melanoma():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image = preprocess_image(file)
        device = torch.device('cpu')
        model = MODEL_FILES['melanoma']['model']
        res = torch.exp(model(image.to(device)))

        classes = {0: 'benign', 1: 'malignant'}
        prediction = classes[res.argmax().item()]

        print("PREDICTION:", prediction)

        # If prediction is benign, then run the other skin model
        if prediction == "benign":
            return predict_skin(image)

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def predict_skin(image):
    device = torch.device('cpu')
    model = MODEL_FILES['dermnet']['model']
    res = torch.exp(model(image.to(device)))

    classes = {
        0: 'Acne and Rosacea Photos',
        1: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
        2: 'Atopic Dermatitis Photos',
        3: 'Bullous Disease Photos',
        4: 'Cellulitis Impetigo and other Bacterial Infections',
        5: 'Eczema Photos',
        6: 'Exanthems and Drug Eruptions',
        7: 'Hair Loss Photos Alopecia and other Hair Diseases',
        8: 'Herpes HPV and other STDs Photos',
        9: 'Light Diseases and Disorders of Pigmentation',
        10: 'Lupus and other Connective Tissue diseases',
        11: 'Melanoma Skin Cancer Nevi and Moles',
        12: 'Nail Fungus and other Nail Disease',
        13: 'Poison Ivy Photos and other Contact Dermatitis',
        14: 'Psoriasis pictures Lichen Planus and related diseases',
        15: 'Scabies Lyme Disease and other Infestations and Bites',
        16: 'Seborrheic Keratoses and other Benign Tumors',
        17: 'Systemic Disease',
        18: 'Tinea Ringworm Candidiasis and other Fungal Infections',
        19: 'Urticaria Hives',
        20: 'Vascular Tumors',
        21: 'Vasculitis Photos',
        22: 'Warts Molluscum and other Viral Infections'
    }

    prediction = classes[res.argmax().item()]
    print("PREDICTION DERMNET:", prediction)

    return jsonify({"prediction": prediction}), 200

# Endpoint for nail predictions
@app.route('/predict/nails', methods=['OPTIONS', 'POST'])
def predict_nails():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image = preprocess_image(file)
        device = torch.device('cpu')
        model = MODEL_FILES['nail']['model']
        res = torch.exp(model(image.to(device)))

        classes = {
            0: 'Darier\'s disease',
            1: 'Muehrcke\'s lines',
            2: 'alopecia areata',
            3: 'Beau\'s lines',
            4: 'bluish nail',
            5: 'clubbing',
            6: 'eczema',
            7: 'half and half nails (Lindsay\'s nails)',
            8: 'koilonychia',
            9: 'leukonychia',
            10: 'onycholysis',
            11: 'pale nail',
            12: 'red lunula',
            13: 'splinter hemorrhage',
            14: 'Terry\'s nails',
            15: 'white nail',
            16: 'yellow nails'
        }

        prediction = classes[res.argmax().item()]
        print("PREDICTION:", prediction)

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint for cataract predictions
@app.route('/predict/cataracts', methods=['OPTIONS', 'POST'])
def predict_cataracts():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image = preprocess_image(file)
        device = torch.device('cpu')
        model = MODEL_FILES['cataract']['model']
        res = torch.exp(model(image.to(device)))

        classes = {0: 'cataract', 1: 'normal'}
        prediction = classes[res.argmax().item()]
        print("PREDICTION:", prediction)

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
