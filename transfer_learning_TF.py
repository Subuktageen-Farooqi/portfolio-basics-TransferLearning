# Step 1 - Import Libraries


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
import numpy as np
import tensorflow as tf
# Step 2 - Load and Preprocess the Image


# Function to load and preprocess the image
def load_and_preprocess_image(img_path, model_name):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224 pixels
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Preprocess image according to the model's requirements
    if model_name == 'vgg':
        return preprocess_vgg(img_array)
    elif model_name == 'resnet':
        return preprocess_resnet(img_array)


# Step 3 - Load Pretrained Models
from tensorflow.keras.applications import VGG16, ResNet50

# Load pretrained VGG16 model
vgg_model = VGG16(weights='imagenet')

# Load pretrained ResNet50 model
resnet_model = ResNet50(weights='imagenet')

# Step 4 - Make Predictions
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_vgg
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_resnet

# Function to make top 5 predictions
def make_predictions(model, img_array, model_name):
    preds = model.predict(img_array)  # Make predictions
    if model_name == 'vgg':
        return decode_vgg(preds, top=5)  # Decode predictions (VGG16 format)
    elif model_name == 'resnet':
        return decode_resnet(preds, top=5)  # Decode predictions (ResNet50 format)

# Predict and get top 5 guesses
def print_top_5_predictions(model, img_path, model_name):
    img_array = load_and_preprocess_image(img_path, model_name)  # Load and preprocess image
    predictions = make_predictions(model, img_array, model_name)  # Get top 5 predictions

    # Print top 5 predictions with probabilities
    print(f"Top 5 predictions for {model_name.upper()} model:")
    for i, pred in enumerate(predictions[0]):
        print(f"{i + 1}. {pred[1]} ({pred[2] * 100:.2f}% probability)")

# Step 5 - Define Image Path
# Define the path to your image
img_path = r"D:\Deep Learning Course\Codes\Images for pretrained\Dog.jpg"

# Step 6 - Get Predictions from VGG16 and ResNet50
# Predict top 5 using VGG16
vgg_predictions = predict_top_5(vgg_model, img_path, 'vgg', preprocess_vgg, decode_vgg)
print("Top 5 predictions from VGG16:")
for i, pred in enumerate(vgg_predictions[0]):
    print(f"{i + 1}. {pred[1]} ({pred[2] * 100:.2f}% probability)")

# Predict top 5 using ResNet50
resnet_predictions = predict_top_5(resnet_model, img_path, 'resnet', preprocess_resnet, decode_resnet)
print("\nTop 5 predictions from ResNet50:")
for i, pred in enumerate(resnet_predictions[0]):
    print(f"{i + 1}. {pred[1]} ({pred[2] * 100:.2f}% probability)") 
