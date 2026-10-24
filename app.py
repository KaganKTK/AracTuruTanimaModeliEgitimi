import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Run the app with:  streamlit run app.py
import torchvision.transforms.functional as F

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
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
 

# --- 1. Sayfa ve Tasarım Ayarları ---
st.set_page_config(page_title="Yapay Zeka Araba Tanıma", page_icon="🚗")
st.title("🚗 Araba Sınıflandırma Modeli")
st.markdown("Colab üzerinde eğittiğimiz derin öğrenme (Transfer Learning) modelini test edin. Bir fotoğraf yükleyin, yapay zeka incelesin!")

# --- 2. Sınıf İsimleri ---
# PyTorch ImageFolder klasörleri okurken alfabetik sıralar. 
# Bu sıralamanın Colab'deki train_dataset.classes çıktısı ile BİREBİR aynı olması şarttır.
CLASSES = ['HATCHBACK', 'MICRO', 'OPEN_WHEEL', 'PICK_UP', 'SEDAN', 'STATION_WAGON', 'SUV', 'VAN']

# --- 3. Eğitilmiş Modeli Yükleme ---
# st.cache_resource: Modelin her fotoğraf yüklendiğinde baştan indirilmesini engeller, hafızada tutar.
@st.cache_resource
def load_model():
    # Modeli kendi bilgisayarımızda (GPU olmama ihtimaline karşı) CPU üzerinde açıyoruz
    device = torch.device('cpu')
    
    # Bu checkpoint EfficientNetV2-S mimarisi ile eğitilmiş.
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 8) # 8 çıkışlı kendi katmanımız
    
    # Kendi eğittiğimiz beyni (pth dosyası) iskelete yerleştir
    # map_location='cpu' parametresi Colab'de GPU ile eğitilen modelin senin bilgisayarında da açılmasını sağlar
    checkpoint = torch.load('efficientnet_araba_modeli.pth', map_location=device, weights_only=True)

    # Bazı eğitim scriptleri modeli "model_state_dict" altında kaydeder.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    model.load_state_dict(checkpoint)
    
    model.eval() # Modeli "Test/Kullanım" moduna geçir (Eğitim modundan çıkar)
    return model

model = load_model()

# --- 4. Görüntü Ön İşleme ---
# Resmi bükmeden ve kırpmadan doğrudan ölçekliyoruz, böylece arabanın tamamı korunuyor.
# app.py — val ile birebir aynı olmalı
transform = transforms.Compose([
    PadToSquare(),           # fill=0 varsayılan, siyah padding
    transforms.Resize(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# --- 5. Kullanıcı Arayüzü (Girdi ve Çıktı) ---
uploaded_file = st.file_uploader("Lütfen bir araba fotoğrafı seçin (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Fotoğrafı ekranda göster
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Yüklediğin Fotoğraf', width='stretch')
    
    if st.button("Türü Tahmin Et 🧠"):
        with st.spinner("Model görüntüyü işliyor..."):
            # Fotoğrafı modele hazır hale getir (unsqueeze ile Batch boyutunu 1 yapıyoruz)
            img_tensor = transform(image).unsqueeze(0)
            
            # Tahmin İşlemi
            with torch.no_grad():
                outputs = model(img_tensor)
                # Modelin verdiği ham skorları (logits), 0-1 arası yüzdelik ihtimallere çevir
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            # Sonuçları Ayrıştır
            predicted_class = CLASSES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            # Ekrana Yazdır
            st.success(f"**Tahmin Edilen Sınıf:** {predicted_class}")
            st.info(f"**Eminlik Oranı:** %{confidence_score:.2f}")
            
            # Analiz Grafiği
            st.write("### Modelin Arka Plandaki Kafa Karışıklığı (Diğer İhtimaller)")
            
            # Pandas Dataframe formatında tüm sınıfların olasılıklarını hazırla
            prob_df = pd.DataFrame({
                "Sınıflar": CLASSES,
                "İhtimal (%)": [probabilities[i].item() * 100 for i in range(len(CLASSES))]
            }).set_index("Sınıflar")
            
            st.bar_chart(prob_df)
