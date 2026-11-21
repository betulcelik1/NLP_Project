# EXP-06 — DeBERTa-v3-small Fine-Tuning (DiCo-NLI Track 1)

## Amaç
SemEval-2027 Task 2 DiCo-NLI Track 1 (İngilizce) için dört sınıflı doğal dil çıkarımı (NLI) görevinde, önceki TF-IDF + Logistic Regression taban çizgisini Transformer tabanlı bir modelle geliştirmek.

## Veri ve ayrım
- Kaynak: [DiCo-NLI resmi veri deposu](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI).
- Eğitim verisi: 3.042 örnek; sınıflar: BACKWARD_ENTAILMENT (881), EQUIVALENCE (802), FORWARD_ENTAILMENT (881), NEGATIVE_OTHER (478).
- Grup bazlı eğitim/doğrulama ayrımı: 2.434 / 608 örnek (yaklaşık %80/%20; random_state=42).
- Aynı `pair_id` ve normalize edilip sırasız karşılaştırılan aynı metin çiftleri, union-find ile aynı gruba alınmış ve `GroupShuffleSplit` kullanılmıştır. Eğitim ve doğrulama arasında ortak `pair_id`: 0; ortak normalize sırasız metin çifti: 0.
- Resmi Dev: 660 etiketli örnek; eğitime dahil edilmemiş, eğitimin ardından değerlendirilmiştir.

## Model ve eğitim yapılandırması
- Model: `microsoft/deberta-v3-small`; dört sınıflı `DebertaV2ForSequenceClassification`.
- Parametre sayısı: 141.897.988 (sınıflandırma katmanları yeni başlatılmıştır).
- Tokenizer: `DebertaV2Tokenizer` (`use_fast=False`); maksimum dizi uzunluğu: 128.
- Epoch: 3; learning rate: 2e-5; train batch size: 8; eval batch size: 16; gradient accumulation: 2 (etkili train batch: 16).
- Weight decay: 0.01; warmup ratio: 0.1; FP16: açık; seed/data_seed: 42.
- Her epoch sonunda değerlendirme ve checkpoint; en iyi checkpoint, validation Weighted F1 ile seçildi.
- Donanım: Tesla T4; PyTorch 2.14.1+cu130, Transformers 4.57.6, SentencePiece 0.2.3.
- Eğitim süresi: 203,60 saniye; 459 adım; en iyi checkpoint: `checkpoint-459` (3. epoch).

## Eğitim sırasında doğrulama sonuçları

| Epoch | Training Loss | Validation Loss | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.1813 | 1.0536 | 0.5921 | 0.4799 | 0.5474 |
| 2 | 0.7040 | 0.7215 | 0.7418 | 0.7062 | 0.7362 |
| 3 | 0.4831 | 0.7117 | **0.7516** | **0.7202** | **0.7486** |

## Resmi Dev sonuçları (660 örnek)

| Metrik | Değer |
|---|---:|
| Accuracy | 0.7045 |
| Macro F1 | 0.6706 |
| **Weighted F1** | **0.6994** |
| Weighted Precision | 0.6974 |
| Weighted Recall | 0.7045 |
| Test Loss | 0.8286 |

### Sınıf bazlı değerlendirme

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BACKWARD_ENTAILMENT | 0.7677 | 0.8000 | 0.7835 | 190 |
| EQUIVALENCE | 0.7143 | 0.6609 | 0.6866 | 174 |
| FORWARD_ENTAILMENT | 0.7311 | 0.8158 | 0.7711 | 190 |
| NEGATIVE_OTHER | 0.4831 | 0.4057 | 0.4410 | 106 |

Confusion matrix (satırlar gerçek sınıf, sütunlar tahmin; sınıf sırası BACKWARD_ENTAILMENT, EQUIVALENCE, FORWARD_ENTAILMENT, NEGATIVE_OTHER):

```text
[[152, 13, 15, 10],
 [ 12,115, 26, 21],
 [  4, 16,155, 15],
 [ 30, 17, 16, 43]]
```

Toplam 465 doğru ve 195 yanlış tahmin.

## Yön tutarlılığı (pair-level)
Resmi Dev içindeki tam iki örnekli `pair_id` gruplarında yön dönüşümü test edildi: FORWARD_ENTAILMENT ↔ BACKWARD_ENTAILMENT; EQUIVALENCE ve NEGATIVE_OTHER sabit.

- Kontrol edilen çift: 277.
- Tutarlı tahmin çifti: 240 / 277 = **0.8664**.
- Her iki örneği de doğru tahmin edilen çift: 196 / 277 = **0.7076**.
- Tutarsız çift: 37.
- Bu ölçümler yalnızca iki örnekli grupları kapsar; tek örnekli gruplar dahil değildir.

## Hata analizi
- En düşük F1, NEGATIVE_OTHER sınıfında (0.4410): 106 örnekten 43'ü doğru, 63'ü yanlış sınıflandırıldı.
- En sık karışıklık: NEGATIVE_OTHER → BACKWARD_ENTAILMENT (30 örnek).
- Diğer belirgin karışıklıklar: EQUIVALENCE → FORWARD_ENTAILMENT (26); EQUIVALENCE → NEGATIVE_OTHER (21); NEGATIVE_OTHER → EQUIVALENCE (17); NEGATIVE_OTHER → FORWARD_ENTAILMENT (16).
- Örnek: `A blonde dog` / `A little blonde girl` gerçekte NEGATIVE_OTHER iken BACKWARD_ENTAILMENT tahmin edildi. Ortak sözcüklere aşırı ağırlık verilmesi olası bir açıklamadır; yalnızca çıktılardan kesin neden saptanamaz.

## Önceki deneylerle karşılaştırma
Aynı **resmi Dev Weighted F1** metriği:

| Deney | Weighted F1 |
|---|---:|
| EXP-01 | 0.4612 |
| EXP-02 | 0.4844 |
| EXP-03 | 0.6151 |
| EXP-04 | 0.6180 |
| EXP-05 | 0.6263 |
| **EXP-06** | **0.6994** |

EXP-05'e göre mutlak artış: **0.0731** (7,31 yüzde puanı). EXP-06'nın ayrılmış validation skoru (0.7486), EXP-05'in resmi Dev skoruyla doğrudan karşılaştırılmamalıdır.

## Kayıtlar ve sonraki adımlar
- Eğitilmiş model ve tokenizer Google Drive'da `MyDrive/NLP_Project/EXP-06_DeBERTa/` klasörüne kaydedildi (`model.safetensors`, `config.json`, tokenizer dosyaları).
- Resmi Dev tahminleri ve metrikleri aynı klasörde `official_dev_predictions.csv` ve `official_dev_metrics.json` olarak oluşturulmak üzere kodlandı; bu rapor dosyaların varlığını ayrıca doğrulamaz.
- Sonraki deneyler: NEGATIVE_OTHER için hata analizi ve sınıf dengesizliği stratejileri; yön tutarlılığını artıran eğitim/değerlendirme yöntemleri. Resmi Dev üzerinde tekrarlı ayar seçiminin aşırı uyuma yol açabileceği göz önünde bulundurulmalıdır.
