import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
"""
Image Classification: Cats vs. Rabbits Using Manual Implementations of ResNet-18 and EfficientNet-B0
This script trains and evaluates two manually implemented models (ResNet-18 and Efficient"""

# 1. CONFIGURATION AND DEVICE SETUP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGE_SIZE = 224
BATCH_SIZE = 32
DATA_ROOT = r"C:\Users\ASUS\Desktop\Deep learning full\dataset\train-cat-rabbit"
TRAIN_RATIO = 0.8
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


train_full_dataset = datasets.ImageFolder(DATA_ROOT, transform=train_transforms)
val_full_dataset = datasets.ImageFolder(DATA_ROOT, transform=val_transforms)

NUM_CLASSES = len(train_full_dataset.classes)
print(f"Detected classes: {train_full_dataset.classes} (NUM_CLASSES={NUM_CLASSES})")

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

print(f"Train images: {len(train_data)} | Val images: {len(val_data)}")


# 4. MANUAL RESNET-18 IMPLEMENTATION


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = self.relu(out)
        return out

class ResNet18Manual(nn.Module):
    """
    Manual ResNet-18:
    conv7x7 -> maxpool -> [2,2,2,2] BasicBlocks -> GAP -> FC.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.in_ch = 64
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)       # 3x224x224 -> 64x112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64x112x112 -> 64x56x56
        # 4 stages: [2,2,2,2] blocks
        self.layer1 = self._make_layer(64, 2, stride=1)    # 64x56x56
        self.layer2 = self._make_layer(128, 2, stride=2)   # 128x28x28
        self.layer3 = self._make_layer(256, 2, stride=2)   # 256x14x14
        self.layer4 = self._make_layer(512, 2, stride=2)   # 512x7x7
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))        # 512x1x1
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)  # 512->2

    def _make_layer(self, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * BasicBlock.expansion),
            )
        layers = []
        layers.append(BasicBlock(self.in_ch, out_ch, stride, downsample))
        self.in_ch = out_ch * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # -> 64x112x112
        x = self.maxpool(x)                    # -> 64x56x56
        x = self.layer1(x)                     # -> 64x56x56
        x = self.layer2(x)                     # -> 128x28x28
        x = self.layer3(x)                     # -> 256x14x14
        x = self.layer4(x)                     # -> 512x7x7
        x = self.avgpool(x)                    # -> 512x1x1
        x = torch.flatten(x, 1)                # -> 512
        x = self.fc(x)                         # -> 2
        return x


# 5. MANUAL EFFICIENTNET-B0 IMPLEMENTATION (SIMPLIFIED EXACT BLOCK CONFIG)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced),
            nn.SiLU(),
            nn.Linear(reduced, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        w = w.view(x.size(0), -1, 1, 1)
        return x * w

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, kernel, stride, use_se=True):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)
        layers = []

        # 1x1 expand (if expand_ratio > 1)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU()
            ]

        # depthwise conv
        layers += [
            nn.Conv2d(hidden, hidden, kernel, stride=stride,
                      padding=kernel // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        ]

        # SE
        if use_se:
            layers += [SEBlock(hidden)]

        # project
        layers += [
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out

class EfficientNetB0Manual(nn.Module):
    """
    Manual EfficientNet-B0 (no pretrained):
    stem -> B0 MBConv config -> head -> GAP -> FC.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # Blocks according to B0 config
        blocks = []
        # (in, out, exp, k, s, repeats)
        settings = [
            # stage1
            [32, 16, 1, 3, 1, 1],
            # stage2
            [16, 24, 6, 3, 2, 2],
            # stage3
            [24, 40, 6, 5, 2, 2],
            # stage4
            [40, 80, 6, 3, 2, 3],
            # stage5
            [80, 112, 6, 5, 1, 3],
            # stage6
            [112, 192, 6, 5, 2, 4],
            # stage7
            [192, 320, 6, 3, 1, 1],
        ]
        for in_ch, out_ch, exp, k, s, r in settings:
            # first block in stage with stride s
            blocks.append(MBConv(in_ch, out_ch, exp, k, s))
            # remaining repeats with stride 1
            for _ in range(r - 1):
                blocks.append(MBConv(out_ch, out_ch, exp, k, 1))

        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)      # -> 32x112x112
        x = self.blocks(x)    # -> 320x7x7 (approx)
        x = self.head(x)      # -> 1280x7x7
        x = self.pool(x)      # -> 1280x1x1
        x = x.view(x.size(0), -1)  # 1280
        x = self.fc(x)        # -> 2
        return x


# 6. MODEL INITIALIZATION


def initialize_model(model_name, num_classes, device):
    if model_name == "ResNet18Manual":
        model = ResNet18Manual(num_classes=num_classes)
    elif model_name == "EfficientNetB0Manual":
        model = EfficientNetB0Manual(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


# 7. EVALUATION FUNCTION


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


# 8. TRAINING LOOP


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

        val_loss, val_acc, val_prec, val_rec, cm, _, _ = \
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

    return model, history, best_cm


# 9. PLOTTING FUNCTIONS


def plot_training_curves(res_hist, eff_hist, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, res_hist['train_loss'], label='ResNet18 Train Loss')
    plt.plot(epochs, res_hist['val_loss'], label='ResNet18 Val Loss')
    plt.plot(epochs, eff_hist['train_loss'], label='EfficientNet-B0 Train Loss')
    plt.plot(epochs, eff_hist['val_loss'], label='EfficientNet-B0 Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, res_hist['train_acc'], label='ResNet18 Train Acc')
    plt.plot(epochs, res_hist['val_acc'], label='ResNet18 Val Acc')
    plt.plot(epochs, eff_hist['train_acc'], label='EfficientNet-B0 Train Acc')
    plt.plot(epochs, eff_hist['val_acc'], label='EfficientNet-B0 Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(res_cm, eff_cm, class_names):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(res_cm, annot=True, fmt="d", cmap="Blues",
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
    plt.show()

def print_metrics_table(
    res_val_loss, res_val_acc, res_prec, res_rec,
    eff_val_loss, eff_val_acc, eff_prec, eff_rec
):
    print("\n===== Final Validation Metrics (Table) =====")
    print(f"{'Model':<18} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'Loss':>10}")
    print("-" * 55)
    print(f"{'ResNet18Manual':<18} {res_val_acc:8.4f} {res_prec:8.4f} {res_rec:8.4f} {res_val_loss:10.4f}")
    print(f"{'EfficientNetB0Manual':<18} {eff_val_acc:8.4f} {eff_prec:8.4f} {eff_rec:8.4f} {eff_val_loss:10.4f}")


# 10. MAIN EXECUTION


if __name__ == "__main__":
    # Initialize models
    resnet_model = initialize_model("ResNet18Manual", NUM_CLASSES, DEVICE)
    efficient_model = initialize_model("EfficientNetB0Manual", NUM_CLASSES, DEVICE)

    # Train ResNet18Manual
    resnet_model, resnet_history, resnet_cm = train_model(
        resnet_model, train_loader, val_loader,
        DEVICE, num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE, model_name="ResNet18Manual"
    )

    # Train EfficientNetB0Manual
    efficient_model, eff_history, eff_cm = train_model(
        efficient_model, train_loader, val_loader,
        DEVICE, num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE, model_name="EfficientNetB0Manual"
    )

    # Final evaluation
    res_val_loss, res_val_acc, res_prec, res_rec, _, _, _ = \
        evaluate_and_collect_metrics(resnet_model, val_loader, DEVICE)
    eff_val_loss, eff_val_acc, eff_prec, eff_rec, _, _, _ = \
        evaluate_and_collect_metrics(efficient_model, val_loader, DEVICE)

    print("\n===== Final Validation Metrics =====")
    print(f"ResNet18Manual      -> Acc: {res_val_acc:.4f}, "
          f"Prec: {res_prec:.4f}, Recall: {res_rec:.4f}, Loss: {res_val_loss:.4f}")
    print(f"EfficientNetB0Manual-> Acc: {eff_val_acc:.4f}, "
          f"Prec: {eff_prec:.4f}, Recall: {eff_rec:.4f}, Loss: {eff_val_loss:.4f}")

    print_metrics_table(
        res_val_loss, res_val_acc, res_prec, res_rec,
        eff_val_loss, eff_val_acc, eff_prec, eff_rec
    )

    # Plots
    plot_training_curves(resnet_history, eff_history, NUM_EPOCHS)

    class_names = train_full_dataset.classes
    plot_confusion_matrices(resnet_cm, eff_cm, class_names)
