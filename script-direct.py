import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Image Classification: Cats vs. Rabbits Using Torchvision Models
This script trains and evaluates two pre-trained models (ResNet18 and EfficientNet-B0)"""

# 1. CONFIGURATION AND DEVICE SETUP


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGE_SIZE = 224          # Input size for ResNet18 / EfficientNet-B0
BATCH_SIZE = 32
DATA_ROOT = r"C:\Users\ASUS\Desktop\Deep learning full\dataset\train-cat-rabbit"
TRAIN_RATIO = 0.8         # 80% train, 20% validation
LEARNING_RATE = 0.001
NUM_EPOCHS = 15

torch.manual_seed(42)


# 2. DATA TRANSFORMS


train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 3. DATA LOADING AND SPLIT


# Load once with each transform
train_full_dataset = datasets.ImageFolder(DATA_ROOT, transform=train_transforms)
val_full_dataset = datasets.ImageFolder(DATA_ROOT, transform=val_transforms)

NUM_CLASSES = len(train_full_dataset.classes)
print(f"Detected classes: {train_full_dataset.classes}  (NUM_CLASSES={NUM_CLASSES})")

full_size = len(train_full_dataset)
train_size = int(TRAIN_RATIO * full_size)
val_size = full_size - train_size

generator = torch.Generator().manual_seed(42)
train_indices, val_indices = random_split(
    range(full_size),
    [train_size, val_size],
    generator=generator
)

train_data = torch.utils.data.Subset(train_full_dataset, train_indices.indices)
val_data = torch.utils.data.Subset(val_full_dataset, val_indices.indices)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4
)

print(f"Train images: {len(train_data)}  |  Val images: {len(val_data)}")


# 4. MODEL INITIALIZATION


def initialize_model(model_name, num_classes, device):
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


# 5. EVALUATION FUNCTION


def evaluate_and_collect_metrics(model, data_loader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds,
                                average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds,
                          average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, cm, all_labels, all_preds


# 6. TRAINING LOOP (GENERIC)


def train_model(model, train_loader, val_loader, device,
                num_epochs=10, lr=1e-3, model_name="model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': []
    }
    best_val_acc = 0.0
    best_state = None
    best_cm = None

    print(f"\n--- Training {model_name} for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        val_loss, val_acc, val_prec, val_rec, cm, v_labels, v_preds = \
            evaluate_and_collect_metrics(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            best_cm = cm

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Prec: {val_prec:.4f} | Recall: {val_rec:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Return best confusion matrix (corresponding to best_val_acc)
    return model, history, best_cm


# 7. PLOTTING FUNCTIONS


def plot_training_curves(resnet_history, eff_history, num_epochs):
    epochs = range(1, num_epochs + 1)

    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, resnet_history['train_loss'], label='ResNet18 Train Loss')
    plt.plot(epochs, resnet_history['val_loss'], label='ResNet18 Val Loss')
    plt.plot(epochs, eff_history['train_loss'], label='EfficientNet-B0 Train Loss')
    plt.plot(epochs, eff_history['val_loss'], label='EfficientNet-B0 Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()  # [web:3]

    # Accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, resnet_history['train_acc'], label='ResNet18 Train Acc')
    plt.plot(epochs, resnet_history['val_acc'], label='ResNet18 Val Acc')
    plt.plot(epochs, eff_history['train_acc'], label='EfficientNet-B0 Train Acc')
    plt.plot(epochs, eff_history['val_acc'], label='EfficientNet-B0 Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()  # [web:1][web:3]

def plot_confusion_matrices(resnet_cm, eff_cm, class_names):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(resnet_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("ResNet18 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.subplot(1, 2, 2)
    sns.heatmap(eff_cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("EfficientNet-B0 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.show()  # [web:6][web:12]

def print_metrics_table(
    res_val_loss, res_val_acc, res_prec, res_rec,
    eff_val_loss, eff_val_acc, eff_prec, eff_rec
):
    print("\n===== Final Validation Metrics (Table) =====")
    print(f"{'Model':<18} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'Loss':>10}")
    print("-" * 55)
    print(f"{'ResNet18':<18} {res_val_acc:8.4f} {res_prec:8.4f} {res_rec:8.4f} {res_val_loss:10.4f}")
    print(f"{'EfficientNet-B0':<18} {eff_val_acc:8.4f} {eff_prec:8.4f} {eff_rec:8.4f} {eff_val_loss:10.4f}")


# 8. MAIN EXECUTION – TRAIN BOTH MODELS AND COMPARE


if __name__ == "_main_":
    # Initialize models
    resnet_model = initialize_model("ResNet18", NUM_CLASSES, DEVICE)
    efficient_model = initialize_model("EfficientNet-B0", NUM_CLASSES, DEVICE)

    # Train ResNet18
    resnet_model, resnet_history, resnet_cm = train_model(
        resnet_model, train_loader, val_loader,
        DEVICE, num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE, model_name="ResNet18"
    )

    # Train EfficientNet-B0
    efficient_model, eff_history, eff_cm = train_model(
        efficient_model, train_loader, val_loader,
        DEVICE, num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE, model_name="EfficientNet-B0"
    )

    # Final evaluation for comparison (on validation set)
    res_val_loss, res_val_acc, res_prec, res_rec, _, _, _ = \
        evaluate_and_collect_metrics(resnet_model, val_loader, DEVICE)
    eff_val_loss, eff_val_acc, eff_prec, eff_rec, _, _, _ = \
        evaluate_and_collect_metrics(efficient_model, val_loader, DEVICE)

    print("\n===== Final Validation Metrics =====")
    print(f"ResNet18       -> Acc: {res_val_acc:.4f}, "
          f"Prec: {res_prec:.4f}, Recall: {res_rec:.4f}, Loss: {res_val_loss:.4f}")
    print(f"EfficientNetB0 -> Acc: {eff_val_acc:.4f}, "
          f"Prec: {eff_prec:.4f}, Recall: {eff_rec:.4f}, Loss: {eff_val_loss:.4f}")

    # Metrics table in console
    print_metrics_table(
        res_val_loss, res_val_acc, res_prec, res_rec,
        eff_val_loss, eff_val_acc, eff_prec, eff_rec
    )

    # Plot training curves
    plot_training_curves(resnet_history, eff_history, NUM_EPOCHS)

    # Plot confusion matrices
    class_names = train_full_dataset.classes
    plot_confusion_matrices(resnet_cm, eff_cm, class_names)