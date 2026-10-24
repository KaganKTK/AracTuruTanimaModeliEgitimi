# 1. Kütüphanelerin Yüklenmesi
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

# Bozuk/Eksik dosyaların yüklenmesine izin ver
ImageFile.LOAD_TRUNCATED_IMAGES = True

print(f"PyTorch Versiyonu: {torch.__version__}")
print(f"Torchvision Versiyonu: {torchvision.__version__}")
print(f"OpenCV Versiyonu: {cv2.__version__}")
# 2. Model Hazırlığı (EfficientNet-V2-S)
model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

print("EfficientNet-V2-S modeli başarıyla yüklendi.")

# GPU kullanılabilirliğini kontrol edelim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model şu an {device} üzerinde çalışıyor.")
# 3. Google Drive Bağlantısı
from google.colab import drive
import os

# Drive'ı bağla
drive.mount('/content/drive')

# Klasör yolunu belirle (Genelde 'MyDrive' altındadır)
dataset_path = '/content/drive/MyDrive/Dataset'

if os.path.exists(dataset_path):
    print("Klasör bulundu. İçindeki bazı dosyalar:")
    files = os.listdir(dataset_path)
    print(files[:10]) # İlk 10 dosyayı göster
else:
    print(f"Hata: {dataset_path} yolu bulunamadı. Lütfen klasör adının doğruluğundan emin olun.")
    import os
import shutil

# Kaynak ve Hedef yollar
drive_path = '/content/drive/MyDrive/Dataset'
local_path = '/content/araba_dataset_hizli'

# Sınıflar ve klasör eşleştirme
categories = ['HATCHBACK', 'MICRO', 'OPEN_WHEEL', 'PICK_UP',
              'SEDAN', 'STATION_WAGON', 'SUV', 'VAN']

drive_to_local_map = {
    'ACIK_TEKERLEKLI': 'OPEN_WHEEL',
    'STATİON_WAGON': 'STATION_WAGON'
}

# Hedef split yapısı
splits = ['train', 'val', 'test']

print("Kopyalama işlemi (Kategori/Split yapısı taranarak) başlıyor...")

# Drive'daki ana dizindeki her bir klasörü (Kategori) gez
for cat_in_drive in os.listdir(drive_path):
    cat_source_path = os.path.join(drive_path, cat_in_drive)

    # Eğer bu bir klasör değilse atla
    if not os.path.isdir(cat_source_path):
        continue

    # Kategori ismini eşleştir
    cat_name = drive_to_local_map.get(cat_in_drive, cat_in_drive)

    if cat_name not in categories:
        continue

    # Her kategori içindeki train, val, test klasörlerini kontrol et
    for split in splits:
        source_dir = os.path.join(cat_source_path, split)

        if os.path.exists(source_dir) and os.path.isdir(source_dir):
            target_dir = os.path.join(local_path, split, cat_name)
            os.makedirs(target_dir, exist_ok=True)

            files = os.listdir(source_dir)
            for f in files:
                src_file = os.path.join(source_dir, f)
                dst_file = os.path.join(target_dir, f)
                if os.path.isfile(src_file):
                    shutil.copy(src_file, dst_file)

            print(f"[{split}] {cat_name}: {len(files)} dosya kopyalandı.")

print(f"\nİşlem tamamlandı. Veriler {local_path} altında PyTorch yapısında (train/val/test) hazır.")
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import os

# --- 1. Transformasyonların Tanımlanması ---
class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        PadToSquare(),
        transforms.Resize(420),
        transforms.RandomCrop(384),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
    ]),
    'val': transforms.Compose([
        PadToSquare(),
        transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# --- 2. Bozuk dosyaları temizle ---
silinen = 0
print("Bozuk dosyalar taranıyor...")
for root, dirs, files in os.walk(local_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except Exception:
                print(f"Bozuk dosya silindi: {file_path}")
                os.remove(file_path)
                silinen += 1
print(f"{silinen} bozuk dosya silindi.\n")

# --- 3. Datasetleri Yükle ---
train_dataset = datasets.ImageFolder(
    root=os.path.join(local_path, 'train'),
    transform=data_transforms['train']
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(local_path, 'val'),
    transform=data_transforms['val']
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(local_path, 'test'),
    transform=data_transforms['val']
)

# --- 4. DataLoader'lar ---
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2),
    'val':   DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2),
}
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# --- 5. Bilgileri Yazdır ---
dataset_sizes = {
    'train': len(train_dataset),
    'val':   len(val_dataset),
    'test':  len(test_dataset)
}
class_names = train_dataset.classes

print(f"Sınıflar: {class_names}")
print(f"Eğitim   : {dataset_sizes['train']} görsel")
print(f"Doğrulama: {dataset_sizes['val']} görsel")
print(f"Test     : {dataset_sizes['test']} görsel")
# 6. Yeni Yapıyı Kontrol Etme
print(f"Dataset yolu: {local_path}\n")
total = 0
for cat in sorted(os.listdir(local_path)):
    cat_path = os.path.join(local_path, cat)
    if os.path.isdir(cat_path):
        count = len(os.listdir(cat_path))
        total += count
        print(f"  - {cat}: {count} fotoğraf")
print(f"\nToplam: {total} fotoğraf")
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import torchvision.transforms.functional as TF

class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        PadToSquare(),
        transforms.Resize(420),
        transforms.RandomCrop(384),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
    ]),
    'val': transforms.Compose([          # 'validation' → 'val'
        PadToSquare(),
        transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

print("Transforms hazır.")              # eski 3 print satırı kaldırıldı
import torchvision.models as models
import torch.nn as nn
import torch

# 8. Modelin 8 Sınıfa Göre Güncellenmesi (EfficientNet-V2-S)

# Modeli tekrar yüklüyoruz (Eğer tanımlı değilse hata almamak için)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

# EfficientNet-V2-S yapısında classifier bir Sequential bloktur ve son katman classifier[1]'dir
num_ftrs = model.classifier[1].in_features

# Çıkış katmanını 8 sınıf olacak şekilde değiştiriyoruz
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

# Modeli GPU'ya gönder
model = model.to(device)

print(f"Model başarıyla {len(class_names)} sınıfa uyarlandı ve {device} üzerine taşındı.")
print(model.classifier)
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Sıralama (ImageFolder alfabetik):
# ACIK_TEKERLEKLI, HATCHBACK, MICRO, PICK_UP, SEDAN, STATİON_WAGON, SUV, VAN
class_weights = torch.tensor(
    [0.83, 0.97, 1.84, 0.98, 0.73, 0.98, 0.89, 1.47], dtype=torch.float
).to(device)

criterion = FocalLoss(alpha=1, gamma=2.0, weight=class_weights)
print("FocalLoss (label_smoothing=0.1) + HATCHBACK ağırlığı (1.2) tanımlandı.")
import time
import torch
import torch.nn as nn

# 9. Eğitim ve Doğrulama Döngüsü
num_ftrs = model.classifier[1].in_features
if model.classifier[1].out_features != len(class_names):
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

num_epochs = 40
patience = 8
epochs_no_improve = 0
best_val_loss = float('inf')
early_stop = False
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

since = time.time()
print(f"Eğitim başlıyor. Sınıf sayısı: {len(class_names)}")
print(f"Kullanılan Kayıp Fonksiyonu: {type(criterion).__name__}\n")

for epoch in range(num_epochs):
    if early_stop: break

    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'train':
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
        else:
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(epoch_acc.item())
            scheduler.step(epoch_loss)
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'en_iyi_araba_modeli.pth')
                print(f"--- Model iyileşti, kaydedildi (Loss: {best_val_loss:.4f}) ---")
            else:
                epochs_no_improve += 1
                print(f"--- İyileşme yok. Sayaç: {epochs_no_improve}/{patience} ---")

            if epochs_no_improve >= patience:
                print("Early Stopping: Eğitim durduruluyor.")
                early_stop = True
    print()

time_elapsed = time.time() - since
print(f'Eğitim {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s içinde tamamlandı.')
import shutil

drive_save_path = '/content/drive/MyDrive/efficientnet_araba_modeli.pth'
shutil.copy('en_iyi_araba_modeli.pth', drive_save_path)

print(f"Model başarıyla Google Drive'a kaydedildi: {drive_save_path}")
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# 11. Test Seti Üzerinde Değerlendirme
all_preds = []
all_labels = []

model.eval()
print("Test analizi başladı...\n")

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

if len(all_preds) > 0:
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.show()
else:
    print("Hata: İşlenecek geçerli resim bulunamadı.")
    import pandas as pd

# 12. Test Sonuçlarını Excel Olarak Kaydetme
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

df_report = pd.DataFrame(report_dict).transpose()

excel_path = '/content/drive/MyDrive/araba_siniflandirma_sonuclari.xlsx'
df_report.to_excel(excel_path, index=True)

print(f"Test sonuçları başarıyla kaydedildi: {excel_path}")
display(df_report)
import matplotlib.pyplot as plt

# 13. Eğitim Geçmişi Görselleştirme
epochs = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_loss'], 'r-o', label='Training Loss')
plt.plot(epochs, history['val_loss'],   'b-o', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_acc'], 'r-s', label='Training Accuracy')
plt.plot(epochs, history['val_acc'],   'b-s', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
