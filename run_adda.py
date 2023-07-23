import argparse
import torch
from adda_mnist_svhn import Encoder, Classifier, Discriminator, load_data, train_adda

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    source_dataloader, target_dataloader, target_test_dataloader, source_test_dataloader = load_data()

    encoder = Encoder().to(device)
    classifier = Classifier().to(device)
    discriminator = Discriminator().to(device)

    learning_curve = train_adda(encoder, classifier, discriminator, source_dataloader, target_dataloader, target_test_dataloader, source_test_dataloader, device, num_epochs=args.epochs)

    print("Training completed.")
    print(f"Best accuracy on source test dataset: {max([acc[0] for acc in learning_curve]):.4f}")
    print(f"Best accuracy on target test dataset: {max([acc[1] for acc in learning_curve]):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ADDA model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computations')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    args = parser.parse_args()
    main(args)
