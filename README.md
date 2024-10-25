# Flask-Service HealthLens Application

This repository deploys the Machine Learning functionality of our HealthLens application. It is through the deployment of this Vercel service that our HealthLens application is able to make quick run-time inferences using one of our four machine learning models (trained in-house) and quickly return a prediction of diagnosis to the user on the application side. Utilizing the compute power of IBM Cloud Notebooks, we trained our four models on publicly-available datasets (linked in the individual notebooks) to allow our service to predict between a grand total of more than 40 conditions, ranging vastly in both condition type and condition severity. This document will be organized by having a general [structure](#structure) layout of the repo, an [installation guide](#installation), and an explanation as to the [context of the larger project](#context-of-larger-project). 

## Structure

The `model_notebooks/` directory contains all the `.ipynb` notebooks used to train the four models we query from the client-side HealthLens application. These were trained using the compute power from IBM Cloud Notebook and we entirely trained on publicly-available datasets on Kaggle (with dataset references in each respective notebook). Each notebook leveraged a pre-trained MobileNet_V2 classifier model, froze its bottom layers during training, and trained (via a fine-tuning schema) a classifier on each of the respective datasets' label classes. Each notebook includes documentation on how we downloaded the datasets, established the label classes, trained the model, and tested the model.

The `*.pth` files are the saved weights for each of the four classifiers we trained. These files are later used in `app.py` to fill an inference model with our saved pre-trained weights and infer a condition based on the image the user of the HealthLens application submits. 

The `vercel.json` file establishes the Vercel service configuration for our specific project. The `requirements.txt` file lists out the specific dependencies required to utilize our service.

The `app.py` file is where we establish an endpoint for our main HealthLens application to call and run inference on user-fed photos. This application handles what model to choose for inference (DermNet, Melanoma Classification, Nail Classification, Cataract Classification) and establishes the downstream logic to feed the prediction as well as accuracy/confidence score of the model back to our application via an API response.

## Installation

Run `pip install -r requirements.txt` to install all the required dependencies for this service. Additionally, access to a GPU will make inference much faster, which is recommended once this service upscales to service many simultaneous users. 

## Context of Larger Project

As mentioned in the header, this repository handles the main machine-learning capabilities of our HealthLens application. The mission of HealthLens is to provide preliminary diagnoses to people unable to reliably access medical expertise and lower the barrier to intentional medical care; the capability that this repository gives to our project is paramount, as without these machine learning capabilities, HealthLens would not be able to provide valuable preliminary diagnoses and would therefore not achieve its mission. While the other repositories of our project (front-end and back-end) are necessary and provide the user with valuable features in both data storage and application usability, it is clear that this machine-learning capabilty is central to our project.

## License and Contributing

Please refer to our [main project repository](https://github.com/ericfly02/Frontend-HealthLens) for the license and contribution information. 