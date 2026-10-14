import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetModel

# ResNet-50 pretrained on ImageNet, as linked by the handout.
MODEL_ID = "microsoft/resnet-50"
IMAGE_SIZE = 224
BATCH_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-limit",
        type=int,
        default=10000,
        help="Number of MNIST training images used to build prototypes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N test images. Use 0 for the full test set.",
    )
    return parser.parse_args()


def prepare_inputs(images, processor, device):
    images = [
        image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        for image in images
    ]
    return processor(
        images=images,
        return_tensors="pt",
        do_resize=False,
    ).to(device)


@torch.inference_mode()
def extract_batch_features(inputs, model, device):
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        output = model(**inputs).pooler_output.flatten(1)
    return F.normalize(output.float(), dim=-1)


@torch.inference_mode()
def extract_features(dataset, processor, model, device):
    features = []
    for start in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[start:start + BATCH_SIZE]
        inputs = prepare_inputs(batch["image"], processor, device)
        features.append(extract_batch_features(inputs, model, device).cpu())
    return torch.cat(features)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained ResNet backbone only. Its ImageNet head is not used,
    # and none of the ResNet weights are updated.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = ResNetModel.from_pretrained(MODEL_ID).to(device).eval()

    train = load_dataset("ylecun/mnist", split="train")
    if args.train_limit > 0:
        train = train.shuffle(seed=42).select(
            range(min(args.train_limit, len(train)))
        )

    train_features = extract_features(train, processor, model, device)
    train_labels = torch.tensor(train["label"])

    # Build one normalized feature prototype for each MNIST digit 0-9.
    prototypes = []
    for digit in range(10):
        digit_features = train_features[train_labels == digit]
        if len(digit_features) == 0:
            raise ValueError(f"No training images found for digit {digit}.")
        prototypes.append(F.normalize(digit_features.mean(dim=0), dim=0))
    prototypes = torch.stack(prototypes)

    test = load_dataset("ylecun/mnist", split="test")
    if args.limit > 0:
        test = test.select(range(min(args.limit, len(test))))

    correct = 0
    total = len(test)
    for start in range(0, total, BATCH_SIZE):
        batch = test[start:start + BATCH_SIZE]
        inputs = prepare_inputs(batch["image"], processor, device)
        features = extract_batch_features(inputs, model, device).cpu()
        predictions = (features @ prototypes.T).argmax(dim=-1)
        labels = torch.tensor(batch["label"])
        correct += (predictions == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Evaluated {total} MNIST test images on {device}.")
    print(f"Frozen ResNet-50, 10-class prototype accuracy: {accuracy:.4f}%")


if __name__ == "__main__":
    main()
