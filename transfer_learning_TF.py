# Step 1 - Import Libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_vgg
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_resnet
import numpy as np


# Step 2 - Load and preprocess image

def load_and_preprocess_image(img_path, model_name):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img)  # To array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

    if model_name == 'vgg':
        img_array = preprocess_vgg(img_array)
    elif model_name == 'resnet':
        img_array = preprocess_resnet(img_array)
    else:
        raise ValueError("model_name must be 'vgg' or 'resnet'")

    return img_array


# Predict top 5 classes

def predict_top_5(model, img_array, model_name):
    preds = model.predict(img_array)

    if model_name == 'vgg':
        return decode_vgg(preds, top=5)
    elif model_name == 'resnet':
        return decode_resnet(preds, top=5)
    else:
        raise ValueError("model_name must be 'vgg' or 'resnet'")


# Print predictions

def print_predictions(predictions, model_name):
    print(f"\nTop 5 predictions from {model_name.upper()}:")
    for i, pred in enumerate(predictions[0]):
        print(f"{i+1}. {pred[1]} ({pred[2]*100:.2f}%)")


def main():
    img_path = input("Enter image path: ").strip()

    # Step 3 - Load Pretrained Models
    vgg_model = VGG16(weights='imagenet')
    resnet_model = ResNet50(weights='imagenet')

    # Step 4 - Make Predictions
    # VGG16 predictions
    vgg_img = load_and_preprocess_image(img_path, 'vgg')
    vgg_preds = predict_top_5(vgg_model, vgg_img, 'vgg')
    print_predictions(vgg_preds, 'vgg16')

    # ResNet50 predictions
    resnet_img = load_and_preprocess_image(img_path, 'resnet')
    resnet_preds = predict_top_5(resnet_model, resnet_img, 'resnet')
    print_predictions(resnet_preds, 'resnet50')


if __name__ == "__main__":
    main()
