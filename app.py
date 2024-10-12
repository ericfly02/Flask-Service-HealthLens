import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torch import nn,optim
from torchvision import transforms, models
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

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

        # returns prediction
        prediction = classes[res.argmax().item()]

        print("PREDICTION:", prediction)


        # if prediction is benign, then run the other skin model
        if prediction == "benign":
            return predict_skin(image)

        # Return the prediction result
        return jsonify({"prediction": prediction}), 200

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
    res = torch.exp(model(image))
    # returns prediction
    prediction = classes[res.argmax().item()]
    print("PREDICTION DERMNET:", prediction)

    return jsonify({"prediction": prediction}), 200



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
        
        # Process the output (e.g., if it's a classification model, take the top prediction)
        # _, predicted = torch.max(output, 1)
        # prediction = predicted.item()

        # returns prediction
        prediction = classes[res.argmax().item()]

        print("PREDICTION:", prediction)

        # Return the prediction result
        return jsonify({"prediction": prediction}), 200

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
        
        # Process the output (e.g., if it's a classification model, take the top prediction)
        # _, predicted = torch.max(output, 1)
        # prediction = predicted.item()

        # returns prediction
        prediction = classes[res.argmax().item()]

        print("PREDICTION:", prediction)

        # Return the prediction result
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
