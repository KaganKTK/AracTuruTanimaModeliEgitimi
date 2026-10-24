# 🚗 Araç Türü Tanıma Modeli

Yüklenilen araç görsellerinin türünü sınıflandırmak için geliştirilen derin öğrenme projesidir. Projede **EfficientNet-V2-S** mimarisi, transfer learning yaklaşımı ve bir **Streamlit** web arayüzü kullanılmıştır.

## ✨ Proje Özellikleri

- Araç görsellerini 8 farklı sınıfa ayırma
- EfficientNet-V2-S tabanlı görüntü sınıflandırma modeli
- GPU varsa GPU, yoksa CPU üzerinde çalışma desteği
- Görselleri kare formata dönüştürerek model girdisine hazırlama
- Streamlit üzerinden görsel yükleme ve tahmin alma
- Tahmin edilen sınıf ve güven oranını gösterme
- Tüm sınıflara ait olasılıkları grafik olarak sunma
- Eğitim sürecinde doğrulama, erken durdurma ve öğrenme oranı azaltma kullanımı
- Test sonuçlarını sınıflandırma raporu ve karmaşıklık matrisi ile değerlendirme

## 🧠 Sınıflar

Model aşağıdaki araç türlerini tanıyacak şekilde eğitilmiştir:

- `HATCHBACK`
- `MICRO`
- `OPEN_WHEEL`
- `PICK_UP`
- `SEDAN`
- `STATION_WAGON`
- `SUV`
- `VAN`

## 📁 Dosya Yapısı

```text
.
├── app.py                  # Streamlit tahmin uygulaması
├── ModelEgitim.py          # Veri hazırlama, model eğitimi ve değerlendirme kodu
├── efficientnet_araba_modeli.pth  # Eğitilmiş model ağırlıkları
└── README.md
```

> `efficientnet_araba_modeli.pth` dosyası modelin çalışması için gereklidir. Büyük dosya boyutu nedeniyle Git deposuna eklenmemişse, eğitim sürecinden sonra oluşturulan model ağırlıklarını proje kök dizinine yerleştirin.

## ⚙️ Gereksinimler

- Python 3.9 veya üzeri
- PyTorch
- Torchvision
- Streamlit
- Pillow
- Pandas
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- Seaborn
- Openpyxl

Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install torch torchvision streamlit pillow pandas numpy opencv-python scikit-learn matplotlib seaborn openpyxl
```

> CUDA destekli bir sistem kullanıyorsanız, PyTorch kurulum komutunu sisteminize uygun şekilde PyTorch resmi kurulum sayfasından seçmeniz önerilir.

## 🚀 Web Uygulamasını Çalıştırma

1. Depoyu klonlayın:

```bash
git clone https://github.com/KaganKTK/AracTuruTanimaModeliEgitimi.git
cd AracTuruTanimaModeliEgitimi
```

2. Eğitilmiş model dosyasının (`efficientnet_araba_modeli.pth`) `app.py` ile aynı klasörde bulunduğundan emin olun.

3. Streamlit uygulamasını başlatın:

```bash
streamlit run app.py
```

4. Tarayıcıda Streamlit tarafından verilen yerel adresi açın.

5. JPG, JPEG veya PNG formatında bir araç görseli yükleyip **Türü Tahmin Et** butonuna basın.

Uygulama, tahmin edilen araç sınıfını, güven oranını ve tüm sınıflara ait olasılıkları gösterir.

## 🏋️ Modeli Eğitme

Model eğitim kodu `ModelEgitim.py` dosyasında yer almaktadır. Bu dosya Google Colab ve Google Drive kullanımına göre hazırlanmıştır.

### Veri kümesi yapısı

Google Drive üzerinde veri kümesinin aşağıdaki yapıda olması beklenir:

```text
Dataset/
├── HATCHBACK/
│   ├── train/
│   ├── val/
│   └── test/
├── MICRO/
│   ├── train/
│   ├── val/
│   └── test/
├── OPEN_WHEEL/
├── PICK_UP/
├── SEDAN/
├── STATİON_WAGON/
├── SUV/
└── VAN/
```

`ModelEgitim.py` dosyası genel olarak şu işlemleri gerçekleştirir:

1. EfficientNet-V2-S modelini ImageNet ağırlıklarıyla yükler.
2. Google Drive'daki veri kümesini yerel çalışma klasörüne kopyalar.
3. Bozuk veya eksik görselleri temizler.
4. Eğitim, doğrulama ve test dönüşümlerini uygular.
5. Modelin son sınıflandırma katmanını 8 sınıfa göre düzenler.
6. Focal Loss, sınıf ağırlıkları ve label smoothing kullanarak modeli eğitir.
7. En iyi modeli `en_iyi_araba_modeli.pth` adıyla kaydeder.
8. Model ağırlıklarını Google Drive'a `efficientnet_araba_modeli.pth` adıyla kopyalar.
9. Test sonuçlarını sınıflandırma raporu ve normalize edilmiş confusion matrix ile değerlendirir.
10. Sonuçları Excel dosyası olarak dışa aktarır.

## 🔍 Görüntü Ön İşleme

Eğitim ve tahmin aşamalarında görseller:

- Kare olacak şekilde padding uygulanarak dönüştürülür.
- `384 × 384` boyutuna getirilir.
- Tensor formatına çevrilir.
- ImageNet ortalama ve standart sapma değerleriyle normalize edilir.

Bu ön işleme adımlarının eğitim ve tahmin sırasında aynı olması, modelin doğru çalışması açısından önemlidir.

## 🛠️ Kullanılan Teknolojiler

- Python
- PyTorch
- Torchvision
- EfficientNet-V2-S
- Streamlit
- OpenCV
- Pillow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab / Google Drive

## ⚠️ Notlar

- `app.py` dosyası varsayılan olarak modeli CPU üzerinde yükler.
- `app.py` içinde tanımlanan sınıf sıralaması, eğitim sırasında kullanılan `ImageFolder` sınıf sıralamasıyla aynı olmalıdır.
- Model ağırlıkları olmadan Streamlit uygulaması tahmin gerçekleştiremez.
- Eğitim kodundaki Google Drive yolları (`/content/drive/MyDrive/Dataset`) kendi veri kümesi konumunuza göre güncellenmelidir.
- Modelin başarımı kullanılan veri kümesinin kalitesine, sınıf dağılımına ve görsellerin çeşitliliğine bağlıdır.

## 📄 Lisans

Bu proje için henüz özel bir lisans belirtilmemiştir. Projeyi kullanmadan, değiştirmeden veya dağıtmadan önce depo sahibiyle iletişime geçmeniz önerilir.

## 👤 Geliştirici

**KaganKTK**

GitHub: [KaganKTK](https://github.com/KaganKTK)
