import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from scripts.prepare_data import prepare_cifar10_datasets
from models.SimpleCNN import SimpleCNN
from models.SimpleViT import SimpleViT
from models.SimpleComposition import SimpleComposition
import pickle



def test_model(model, test_loader, device):
    # TODO: Implement your test procedure.
    # raise NotImplementedError()
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1)==y).sum().item()
            total += x.size(0)
    accuracy = 100 * correct / total
    
    return accuracy

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="SimpleCNN", type=str)
    args = parser.parse_args()

    # TODO: Set your device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Prepare datasets
    _, _, test_dataset = prepare_cifar10_datasets(None)

    # TODO: Prepare data loader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model_ckpt_save_path = f"./model_ckpt/cifar10_{args.model}_arg.pickle"
    # TODO: Load `model_hps`` with pickle from `model_ckpt_save_path`.
    with open(model_ckpt_save_path, "rb") as f:
        model_hps = pickle.load(f)
    
    if args.model == "SimpleCNN":
        model = SimpleCNN(**model_hps)
    elif args.model == "SimpleViT":
        model = SimpleViT(**model_hps)
    elif args.model == "SimpleComposition":
        model = SimpleComposition(model_hps["CNN_model_hps"], model_hps["ViT_model_hps"])
    else:
        raise ValueError(f"Model {args.model} is not implemented!")
    
    model_save_path = f"./model_ckpt/cifar10_{model.name()}.pth"
    # TODO: Load model state_dicts from `model_save_path`.
    # raise NotImplementedError()
    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict)

    # Test the model
    test_accuracy = test_model(model, test_loader, device)

    print(f'Test Accuracy of the model on the CIFAR-10 test dataset in %: {test_accuracy:.2f}')
