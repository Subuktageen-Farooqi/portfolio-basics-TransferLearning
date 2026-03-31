import torch
from torchvision import models
from torchvision.models import AlexNet_Weights, ResNet101_Weights, MobileNet_V2_Weights
from PIL import Image


def load_image(image_path: str, transform):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


def top5_predictions(model, inputs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, k=5)

    return [(labels[idx], prob.item()) for idx, prob in zip(top_idxs, top_probs)]


def main():
    image_path = input("Enter image path: ").strip()

    experiments = [
        ("AlexNet", AlexNet_Weights.DEFAULT, models.alexnet),
        ("ResNet101", ResNet101_Weights.DEFAULT, models.resnet101),
        ("MobileNetV2", MobileNet_V2_Weights.DEFAULT, models.mobilenet_v2),
    ]

    all_results = {}
    for name, weights, model_builder in experiments:
        model = model_builder(weights=weights)
        inputs = load_image(image_path, weights.transforms())
        labels = weights.meta["categories"]

        preds = top5_predictions(model, inputs, labels)
        all_results[name] = preds

        print(f"\nTop 5 predictions for {name}:")
        for i, (label, prob) in enumerate(preds, start=1):
            print(f"{i}. {label} ({prob * 100:.2f}% probability)")

    print("\nComparison summary:")
    top1 = {name: preds[0] for name, preds in all_results.items()}
    for model_name, (label, prob) in top1.items():
        print(f"- {model_name} top-1: {label} ({prob * 100:.2f}%)")

    unique_top1 = len(set(label for label, _ in top1.values()))
    if unique_top1 == 1:
        print("- All three architectures agree on the same top-1 class.")
    else:
        print("- The architectures disagree on top-1 class, showing architecture-dependent behavior.")


if __name__ == "__main__":
    main()
