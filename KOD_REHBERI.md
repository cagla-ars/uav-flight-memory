# Kod Rehberi — UAV Flight Memory

Bu belge projeyi öğrenmek için kısa bir yol haritasıdır. Kodun çalışması için gereken modüller özellikle korunmuştur.

## Projenin akışı

1. `data/mapping.py`: Uçuş kimliklerini, sınıf etiketlerini ve anotasyon aralıklarını okur.
2. `data/download.py`: Uçuş CSV dosyasının yerel yolunu bulur; yoksa veri setinden indirmeyi dener.
3. `data/quality.py`: Gerekli sinyalleri ve CSV veri kalitesini kontrol eder.
4. `labeling/intervals.py`: Anomali zaman aralıklarını birleştirir ve pencereyle örtüşme süresini hesaplar.
5. `features/windowing.py`: Uçuşu 20 saniyelik pencerelere ayırır, etiket belirler ve 24 özellik çıkarır.
6. `experiments/splitting.py`: Eğitim ve test kümelerini uçuş kimliğine göre ayırır.
7. `models/knn.py`: Eksik değerleri doldurur, özellikleri ölçeklendirir, KNN tahmini yapar ve metrikleri hesaplar.
8. `models/memory.py`: Tahmin edilen pencereye en yakın eğitim pencerelerini bulur.
9. `experiments/runner.py`: Veri hazırlama ve deney adımlarını sırayla çalıştırır.
10. `web/app.py` ve `web/static/`: Python API'si ve tarayıcı arayüzüdür.

## Önce öğrenilecek dosyalar

Önerilen sıra:

1. `config.py` — ortak ayarlar
2. `features/windowing.py` — 20 saniyelik pencereler ve 24 özellik
3. `experiments/splitting.py` — uçuş bazlı eğitim/test ayrımı
4. `models/knn.py` — KNN, ölçeklendirme ve metrikler
5. `models/memory.py` — benzer geçmiş pencereler
6. `experiments/runner.py` — modülleri birleştiren deney akışı
7. `web/app.py` ve `web/static/app.js` — sonuçların gösterimi

## Sık kullanılan ayarlar

- Altı sinyal: `x`, `y`, `z`, `vx`, `vy`, `vz`
- Her sinyal için dört istatistik: `mean`, `std`, `min`, `max`
- Toplam özellik: 6 × 4 = 24
- Pencere: 20 saniye
- K değerleri: 1, 3, 5, 7, 9
- Uzaklıklar: Euclidean ve Manhattan
- Eğitim/test oranları: 80/20 ve 90/10
- Rastgelelik tohumu: 42

## Dosyaları neden rastgele silmiyoruz?

Bazı modüller sunum sırasında doğrudan görünmese de ana kod tarafından çağrılır. Örneğin `intervals.py` etiketlemeye, `splitting.py` deneylerin doğru ayrılmasına, `memory.py` geçmiş uçuş tablosuna hizmet eder. Bunları silmek projeyi bozabilir.

## Testleri çalıştırma

Proje kök klasöründe sanal ortam ve bağımlılıklar kurulduktan sonra:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Testler kaynak kodun bazı temel davranışlarını kontrol eder. Testlerin geçmesi, bütün veri indirme ve web dağıtım senaryolarının da otomatik olarak doğrulandığı anlamına gelmez.
