import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet18_Weights
import timm

def get_class_weights(dataset, num_classes):
    class_counts = np.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1
    class_weights = 1 / class_counts
    return class_weights

def get_sample_weights(dataset, class_weights):
    sample_weights = np.zeros(len(dataset))
    for idx, (_, label) in enumerate(dataset):
        sample_weights[idx] = class_weights[label]
    return sample_weights

def load_data():    
    transform_svhn_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])

    transform_mnist_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])

    transform_svhn_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])

    transform_mnist_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])


    svhn_train = datasets.SVHN(root='./data', split='train', transform=transform_svhn_train, download=True)
    svhn_test = datasets.SVHN(root='./data', split='test', transform=transform_svhn_test, download=True)
    mnist_train = datasets.MNIST(root='./data', train=True, transform=transform_mnist_train, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, transform=transform_mnist_test, download=True)
  
    # Get class weights and sample weights for the source domain dataset
    svhn_class_weights = get_class_weights(svhn_train, 10)
    svhn_sample_weights = get_sample_weights(svhn_train, svhn_class_weights)

    # Create a WeightedRandomSampler for the source domain dataset
    source_sampler = WeightedRandomSampler(weights=svhn_sample_weights, num_samples=len(svhn_sample_weights), replacement=True)

    # Create DataLoaders
    source_dataloader = DataLoader(svhn_train, batch_size=64, sampler=source_sampler)
    source_test_dataloader = DataLoader(svhn_test, batch_size=64, shuffle=False)
    target_dataloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    target_test_dataloader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    return source_dataloader, target_dataloader, target_test_dataloader, source_test_dataloader

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet18 = timm.create_model("resnet18", pretrained=True, num_classes=0)

    def forward(self, x):
        return self.resnet18(x)

class Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(Classifier, self).__init__()
        #self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        #x = self.dropout(x)
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.fc(x)

def train_adda(encoder, classifier, discriminator, source_dataloader, target_dataloader, target_test_dataloader, source_test_dataloader, device, num_epochs=100):
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    discriminator_criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Learning rate schedulers
    classifier_scheduler = StepLR(classifier_optimizer, step_size=15, gamma=0.5)
    encoder_scheduler = StepLR(encoder_optimizer, step_size=15, gamma=0.5)
    discriminator_scheduler = StepLR(discriminator_optimizer, step_size=15, gamma=0.5)

    best_source_accuracy = 0.0
    best_target_accuracy = 0.0
    learning_curve = []

    for epoch in range(num_epochs):
        encoder.train()
        classifier.train()
        discriminator.train()

        epoch_classification_losses = []
        epoch_discriminator_losses = []
        epoch_encoder_losses = []

        # Train on source domain
        for images, labels in tqdm(source_dataloader, desc=f"Epoch {epoch+1} - Source"):
            images, labels = images.to(device), labels.to(device)

            classifier_optimizer.zero_grad()
            encoder_optimizer.zero_grad()

            features = encoder(images)
            preds = classifier(features)
            classification_loss = classification_criterion(preds, labels)
            classification_loss.backward()

            classifier_optimizer.step()
            encoder_optimizer.step()

            epoch_classification_losses.append(classification_loss.item())

        # Train on target domain
        for target_images, _ in tqdm(target_dataloader, desc=f"Epoch {epoch+1} - Target"):
            target_images = target_images.to(device)

            discriminator_optimizer.zero_grad()
            encoder_optimizer.zero_grad()

            target_features = encoder(target_images)

            # Train the discriminator
            source_labels = torch.ones(features.size(0), 1).to(device)
            target_labels = torch.zeros(target_features.size(0), 1).to(device)
            all_features = torch.cat((features, target_features), 0)
            all_labels = torch.cat((source_labels, target_labels), 0)

            discriminator_preds = discriminator(all_features.detach())
            discriminator_loss = discriminator_criterion(discriminator_preds, all_labels)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            epoch_discriminator_losses.append(discriminator_loss.item())

            # Train the encoder (adversarial training)
            target_preds = discriminator(target_features)
            encoder_loss = -discriminator_criterion(target_preds, target_labels)
            encoder_loss.backward()
            encoder_optimizer.step()

            epoch_encoder_losses.append(encoder_loss.item())

        # Calculate average losses for the current epoch
        avg_classification_loss = np.mean(epoch_classification_losses)
        avg_discriminator_loss = np.mean(epoch_discriminator_losses)
        avg_encoder_loss = np.mean(epoch_encoder_losses)
        
        # Update learning rate schedulers
        classifier_scheduler.step()
        encoder_scheduler.step()
        discriminator_scheduler.step()

        # Evaluate the model on the source test dataset
        encoder.eval()
        classifier.eval()

        source_correct = 0
        source_total = 0

        with torch.no_grad():
            for images, labels in source_test_dataloader:
                images, labels = images.to(device), labels.to(device)
                features = encoder(images)
                preds = classifier(features)
                _, predicted = torch.max(preds.data, 1)
                source_total += labels.size(0)
                source_correct += (predicted == labels).sum().item()

        source_accuracy = source_correct / source_total
        print(f"Epoch {epoch+1} - Accuracy on source test dataset: {source_accuracy:.4f}")

        # Evaluate the model on the target test dataset
        target_correct = 0
        target_total = 0

        with torch.no_grad():
            for images, labels in target_test_dataloader:
                images, labels = images.to(device), labels.to(device)
                features = encoder(images)
                preds = classifier(features)
                _, predicted = torch.max(preds.data, 1)
                target_total += labels.size(0)
                target_correct += (predicted == labels).sum().item()

        target_accuracy = target_correct / target_total
        print(f"Epoch {epoch+1} - Accuracy on target test dataset: {target_accuracy:.4f}")

        learning_curve.append((source_accuracy, target_accuracy, avg_classification_loss, avg_discriminator_loss, avg_encoder_loss))

        # Save the best model for source domain
        if source_accuracy > best_source_accuracy:
            best_source_accuracy = source_accuracy
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }, "best_source_model.pth")

        # Save the best model for target domain
        if target_accuracy > best_target_accuracy:
            best_target_accuracy = target_accuracy
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }, "best_target_model.pth")
            
    torch.save({'learning_curve': learning_curve}, 'learning_curve.pth')

    return learning_curve
