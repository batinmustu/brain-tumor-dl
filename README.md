# ResNet152V2 Kullanarak Beyin Tümörü Sınıflandırması

## Genel Bakış

Bu proje, beyin tümörü görüntülerini dört kategoriye sınıflandırmak için ResNet152V2 mimarisi tabanlı bir Konvolüsyonel Sinir Ağı (CNN) kullanır:
- Glioma Tümörü
- Tümör Yok
- Meningioma Tümörü
- Hipofiz Tümörü

Eğitim ve değerlendirme için kullanılan veri seti, Kaggle'dan indirilebilir: [Beyin Tümörü Sınıflandırması MRI](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri).

## Veri Seti

Veri seti iki ana klasöre ayrılmıştır:
- `Training/`: Tümör türüne göre kategorize edilmiş eğitim görüntülerini içerir.
- `Testing/`: Tümör türüne göre kategorize edilmiş test görüntülerini içerir.

Veri setini indirdikten sonra, `archive` adlı bir klasör oluşturup veri setini bu klasörün içine yerleştirin. Kodun bu klasör yapısına uygun olarak çalışması gerekmektedir.

## Gereksinimler

Aşağıdaki Python kütüphanelerinin kurulması gerekmektedir:

- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`
- `opencv-python`
- `tensorflow`
- `tqdm`
- `Pillow`
- `scikit-learn`

Bu kütüphaneleri kurmak için şu komutu kullanabilirsiniz:

```bash
pip install matplotlib numpy pandas seaborn opencv-python tensorflow tqdm Pillow scikit-learn
```
## Model Eğitimi

Model, ResNet152V2 mimarisi üzerine inşa edilmiştir. Eğitilmiş olan model, kategorik çapraz entropi (categorical_crossentropy) kaybı ve Adam optimizasyon algoritması kullanılarak derlenmiştir. Modelin performansını artırmak için erken durdurma (EarlyStopping), öğrenme oranı azaltma (ReduceLROnPlateau) ve en iyi modeli kaydetme (ModelCheckpoint) gibi geri çağrım yöntemleri kullanılmıştır.

Eğitim süreci, aşağıdaki adımları içerir:

1. Eğitim verisinin yüklenmesi ve yeniden boyutlandırılması.
2. Verinin artırılması (ImageDataGenerator) ve eğitim/test verisine ayrılması.
3. ResNet152V2 modelinin eğitilmesi.
4. Eğitim ve doğrulama kayıplarının ve doğruluğunun görselleştirilmesi.

## Model Değerlendirmesi

Eğitim tamamlandıktan sonra, model test verisi üzerinde değerlendirilir ve sınıflandırma raporu `(classification_report)` ve karışıklık matrisi `(confusion_matrix)` ile sonuçlar görselleştirilir.

## Görselleştirme

Eğitim süreci ve modelin performansı, aşağıdaki yöntemlerle görselleştirilmiştir:

- Eğitim ve doğrulama doğruluğu ve kaybı grafikleri.
- Karışıklık matrisi ısı haritası.
- Bu grafikleri görmek için kodu çalıştırdıktan sonra üretilen görselleri inceleyebilirsiniz.

## Sonuçlar

Eğitim tamamlandığında, modelin performansını değerlendiren rapor ve karışıklık matrisi görüntülenecektir. Bu çıktılar, modelin hangi tümör türlerini doğru sınıflandırdığını ve hangi türlerde hata yaptığını gösterir.

## Katkıda Bulunma

Herhangi bir hata bulursanız ya da geliştirme önerileriniz varsa, lütfen bir sorun bildirimi (issue) açın ya da bir çekme isteği (pull request) gönderin.

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Ayrıntılar için `LICENSE` dosyasına bakabilirsiniz.
