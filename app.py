import os
import io
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torch import nn,optim
from torchvision import transforms, models
from flask_cors import CORS
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv
import subprocess
import requests
import tempfile

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Watson Speech to Text API credentials
API_KEY = os.getenv('IBM_SPEECH_TO_TEXT_API_KEY')
API_URL = os.getenv('IBM_SPEECH_TO_TEXT_URL')

# Setup the IBM Watson Speech to Text service
authenticator = IAMAuthenticator(API_KEY)
speech_to_text = SpeechToTextV1(authenticator=authenticator)
speech_to_text.set_service_url(API_URL)

# Endpoint to upload an image and get predictions
@app.route('/predict/skin', methods=['POST'])
def predict_melanoma():
    # Load the trained model and map to CPU
    MODEL_PATH = "melanoma_classifier_10_epochs.pth"  # Path to your .pth model, should be an argument later or part of the request
    device = torch.device('cpu')  # Use CPU

    # load the model with the correct parameter architecture
    model = models.mobilenet_v2(weights='DEFAULT')
    # NOTE: This classifier architecture is SPECIFIC TO THE MELANOMA CLASSIFIER
    model.classifier= nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                    nn.Linear(in_features=1280, out_features=2, bias=True),
                                    nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Load model on CPU
    model.to(device) # ads to device
    model.eval()  # Set model to evaluation mode

    # Define image transformations (adjust these according to your model's requirements)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the expected input size
        transforms.ToTensor(),           # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (if required)
    ])

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Open the image file
        image = Image.open(file).convert('RGB')
        
        # Preprocess the image and convert it to a tensor
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run the image through the model
        # with torch.no_grad():
        #     output = model(image)
        # result
        res = torch.exp(model(image))

        # define classes for well-defined prediction output
        # NOTE: SPECIFIC TO MELANOMA CLASSIFER
        classes = {'benign': 0, 'malignant': 1}
        classes = {value:key for key, value in classes.items()} # invert for proper argmax
        
        # Process the output (e.g., if it's a classification model, take the top prediction)
        # _, predicted = torch.max(output, 1)
        # prediction = predicted.item()

        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(image)  # Get the log-probabilities
            probabilities = torch.exp(output)  # Convert log-probabilities to probabilities

        # Get the predicted class and its confidence
        confidence, predicted_class_idx = torch.max(probabilities, dim=1)
        predicted_class = classes[predicted_class_idx.item()]
        
        # Convert the confidence tensor to a Python float
        confidence_score = confidence.item()

        print(f"PREDICTION: {predicted_class}, CONFIDENCE: {confidence_score:.4f}")

        # if prediction is benign, then run the other skin model
        if predicted_class == "benign":
            return predict_skin(image)

        return jsonify({"prediction": predicted_class, "confidence": confidence_score}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

### DEFINE PREDICT SKINS
def predict_skin(image):
        # Load the trained model and map to CPU
    MODEL_PATH = "dermnet_classifier_10_epochs.pth"  # Path to your .pth model, should be an argument later or part of the request
    device = torch.device('cpu')  # Use CPU

    # load the model with the correct parameter architecture
    model = models.mobilenet_v2(weights='DEFAULT')
    # NOTE: This classifier architecture is SPECIFIC TO THE MELANOMA CLASSIFIER
    model.classifier= nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                    nn.Linear(in_features=1280, out_features=23, bias=True),
                                    nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Load model on CPU
    model.to(device) # ads to device
    model.eval()  # Set model to evaluation mode

    # Define image transformations (adjust these according to your model's requirements)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the expected input size
        transforms.ToTensor(),           # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (if required)
    ])

    classes = {'Acne and Rosacea Photos': 0, 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 1, 'Atopic Dermatitis Photos': 2, 'Bullous Disease Photos': 3, 'Cellulitis Impetigo and other Bacterial Infections': 4, 
    'Eczema Photos': 5, 'Exanthems and Drug Eruptions': 6, 'Hair Loss Photos Alopecia and other Hair Diseases': 7, 'Herpes HPV and other STDs Photos': 8, 'Light Diseases and Disorders of Pigmentation': 9, 'Lupus and other Connective Tissue diseases': 10, 
    'Melanoma Skin Cancer Nevi and Moles': 11, 'Nail Fungus and other Nail Disease': 12, 'Poison Ivy Photos and other Contact Dermatitis': 13, 'Psoriasis pictures Lichen Planus and related diseases': 14, 'Scabies Lyme Disease and other Infestations and Bites': 15, 
    'Seborrheic Keratoses and other Benign Tumors': 16, 'Systemic Disease': 17, 'Tinea Ringworm Candidiasis and other Fungal Infections': 18, 'Urticaria Hives': 19, 'Vascular Tumors': 20, 'Vasculitis Photos': 21, 'Warts Molluscum and other Viral Infections': 22}
    classes = {value:key for key, value in classes.items()} # invert for proper argmax

    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(image)  # Get the log-probabilities
        probabilities = torch.exp(output)  # Convert log-probabilities to probabilities

    # Get the predicted class and its confidence
    confidence, predicted_class_idx = torch.max(probabilities, dim=1)
    predicted_class = classes[predicted_class_idx.item()]
    
    # Convert the confidence tensor to a Python float
    confidence_score = confidence.item()

    print(f"PREDICTION: {predicted_class}, CONFIDENCE: {confidence_score:.4f}")

    return jsonify({"prediction": predicted_class, "confidence": confidence_score}), 200

### ROUTE FOR DERMNET PREDICTIONS:
# Endpoint to upload an image and get predictions
@app.route('/predict/nails', methods=['POST'])
def predict_nails():
    # Load the trained model and map to CPU
    MODEL_PATH = "nail_classifier_10_epochs.pth"  # Path to your .pth model, should be an argument later or part of the request
    device = torch.device('cpu')  # Use CPU

    # load the model with the correct parameter architecture
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier= nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                    nn.Linear(in_features=1280, out_features=17, bias=True),
                                    nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Load model on CPU
    model.to(device) # ads to device
    model.eval()  # Set model to evaluation mode

    # Define image transformations (adjust these according to your model's requirements)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the expected input size
        transforms.ToTensor(),           # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (if required)
    ])

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Open the image file
        image = Image.open(file).convert('RGB')
        
        # Preprocess the image and convert it to a tensor
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run the image through the model
        # with torch.no_grad():
        #     output = model(image)
        # result
        res = torch.exp(model(image))

        # define classes for well-defined prediction output
        # NOTE: SPECIFIC TO MELANOMA CLASSIFER
        classes = {'Darier_s disease': 0, 'Muehrck-e_s lines': 1, 'aloperia areata': 2, 'beau_s lines': 3, 'bluish nail': 4, 'clubbing': 5, 'eczema': 6, 'half and half nailes (Lindsay_s nails)': 7,
         'koilonychia': 8, 'leukonychia': 9, 'onycholycis': 10, 'pale nail': 11, 'red lunula': 12, 'splinter hemmorrage': 13, 'terry_s nail': 14, 'white nail': 15, 'yellow nails': 16}
        classes = {value:key for key, value in classes.items()} # invert for proper argmax
        
        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(image)  # Get the log-probabilities
            probabilities = torch.exp(output)  # Convert log-probabilities to probabilities

        # Get the predicted class and its confidence
        confidence, predicted_class_idx = torch.max(probabilities, dim=1)
        predicted_class = classes[predicted_class_idx.item()]
        
        # Convert the confidence tensor to a Python float
        confidence_score = confidence.item()

        print(f"PREDICTION: {predicted_class}, CONFIDENCE: {confidence_score:.4f}")

        return jsonify({"prediction": predicted_class, "confidence": confidence_score}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


### FOR CATARACTS
# Endpoint to upload an image and get predictions
@app.route('/predict/cataracts', methods=['POST'])
def predict_cataracts():
    # Load the trained model and map to CPU
    MODEL_PATH = "cataract_classifier_10_epochs.pth"  # Path to your .pth model, should be an argument later or part of the request
    device = torch.device('cpu')  # Use CPU

    # load the model with the correct parameter architecture
    model = models.mobilenet_v2(weights='DEFAULT')
    # NOTE: This classifier architecture is SPECIFIC TO THE MELANOMA CLASSIFIER
    model.classifier= nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                    nn.Linear(in_features=1280, out_features=2, bias=True),
                                    nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Load model on CPU
    model.to(device) # ads to device
    model.eval()  # Set model to evaluation mode

    # Define image transformations (adjust these according to your model's requirements)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the expected input size
        transforms.ToTensor(),           # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (if required)
    ])

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Open the image file
        image = Image.open(file).convert('RGB')
        
        # Preprocess the image and convert it to a tensor
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run the image through the model
        # with torch.no_grad():
        #     output = model(image)
        # result
        res = torch.exp(model(image))

        # define classes for well-defined prediction output
        # NOTE: SPECIFIC TO MELANOMA CLASSIFER
        classes = {'cataract': 0, 'normal': 1}
        classes = {value:key for key, value in classes.items()} # invert for proper argmax
        
        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(image)  # Get the log-probabilities
            probabilities = torch.exp(output)  # Convert log-probabilities to probabilities

        # Get the predicted class and its confidence
        confidence, predicted_class_idx = torch.max(probabilities, dim=1)
        predicted_class = classes[predicted_class_idx.item()]
        
        # Convert the confidence tensor to a Python float
        confidence_score = confidence.item()

        print(f"PREDICTION: {predicted_class}, CONFIDENCE: {confidence_score:.4f}")

        return jsonify({"prediction": predicted_class, "confidence": confidence_score}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Convert audio file to PCM if needed
def convert_audio_to_wav(input_audio_path):
    output_file = "converted_audio.wav"
    try:
        command = [
            "ffmpeg", "-i", input_audio_path, 
            "-ar", "16000",  # Set sample rate to 16000 Hz
            "-ac", "1",      # Set number of audio channels to 1 (mono)
            "-f", "wav",     # Output format as WAV
            "-acodec", "pcm_s16le",  # Codec: PCM 16-bit little endian
            output_file
        ]
        subprocess.run(command, check=True)
        return output_file
    except Exception as e:
        raise RuntimeError(f"Failed to convert audio: {str(e)}")


### FOR IBM SPEECH-TO-TEXT-SERVICE
# Endpoint to upload to ibm speech-to-text service the audio
@app.route('/speech/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    # Save the uploaded audio file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        audio_file.save(temp_audio.name)
    
    try:
        # Convert the audio to WAV format compatible with IBM
        converted_audio = convert_audio_to_wav(temp_audio.name)
        
        with open(converted_audio, 'rb') as wav_file:
            # Send the converted audio file to IBM Speech to Text
            response = speech_to_text.recognize(
                audio=wav_file,
                content_type='audio/wav'  # This should now be correct
            ).get_result()

            # Check if transcription results are available
            if response.get('results') and len(response['results']) > 0:
                # Extract transcription if available
                transcription = response['results'][0]['alternatives'][0].get('transcript', "")
                return jsonify({"transcription": transcription}), 200
            else:
                # If no results are available, return an appropriate message
                return jsonify({"error": "No transcription available"}), 204

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup temporary and converted files
        os.remove(temp_audio.name)
        os.remove(converted_audio)




if __name__ == '__main__':
    pass
