import torch
from torchvision import models
from torchvision.models import VGG16_Weights, ResNet50_Weights
from PIL import Image


def load_image(image_path: str, transform):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


def print_top5(model, inputs, labels, model_name: str):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, k=5)

    print(f"\nTop 5 predictions for {model_name}:")
    for rank, (idx, prob) in enumerate(zip(top_idxs.tolist(), top_probs.tolist()), start=1):
        print(f"{rank}. {labels[idx]} ({prob * 100:.2f}% probability)")


def main():
    image_path = input("Enter image path: ").strip()

    vgg_weights = VGG16_Weights.DEFAULT
    resnet_weights = ResNet50_Weights.DEFAULT

    preprocess = vgg_weights.transforms()
    labels = vgg_weights.meta["categories"]

    inputs = load_image(image_path, preprocess)

    vgg16 = models.vgg16(weights=vgg_weights)
    resnet50 = models.resnet50(weights=resnet_weights)

    print_top5(vgg16, inputs, labels, "VGG16")
    print_top5(resnet50, inputs, labels, "ResNet50")


if __name__ == "__main__":
    main()
