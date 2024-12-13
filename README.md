# Speech-to-Text (STT) and Named Entity Recognition (NER) Pipeline for Uzbek Language

## **Project Overview**
Ushbu loyiha O'zbek tili uchun STT va shundan keyin matnni tahlil qilib, NER vazifasini bajaradi. Loyihada fine-tuning jarayonlari amalga oshirildi va pipeline tuzildi.

**Yakuniy Natija**

- **STT**: WER ~30
- **NER**: Precision ~0.97

**Model in Huggingface**

- **STT Model**: ![STT Screenshot](https://github.com/user-attachments/assets/6905c1dd-1e64-4e94-849f-af82f9efd66f)
- **NER Model**: ![NER Screenshot](https://github.com/user-attachments/assets/2b71a3bf-8d09-493c-ba04-5927247fab6e)

---

## **STT Model Details**

### **Main**
Resurslarning cheklanganligi tufayli Whisper-base modeli umumiy datasetning kichik qismi uchun trening qilindi. Birinchi modeldan WER ~70 natijasi olindi. Trening jarayonining davomiyligi 2 soat.

Model keyingi kichik dataset uchun qayta trening qilinib, WER ~32 natijasi olindi. Trening jarayoni 2 soat 40 minut davom etdi.

**Muammo**: Cheklangan resurslar (disk, GPU).

**Yechim**: Kaggle kabi bepul resurslardan foydalanib, treningni ikki bosqichda o'tkazdim:

```
whisper-base -> whisper-uz -> whisper-uz-v2
```

### **Models:**

- Base: [openai/whisper-base](https://huggingface.co/openai/whisper-base)
- Pre-trained v1: [jamshidahmadov/whisper-uz](https://huggingface.co/jamshidahmadov/whisper-uz)
- Pre-trained v2: [jamshidahmadov/whisper-uz-v2](https://huggingface.co/jamshidahmadov/whisper-uz-v2)

### **Notebook**
Trening uchun [notebook](https://github.com/jamshid-ds/uzbek-stt-ner/tree/main/STT/Training).

### **Dataset**

- Asosiy dataset: [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- Whisper-uz-v1 uchun: 10,000 audio
- Whisper-uz-v2 uchun: 21,000 audio

### **Basic Hyperparameters**

- Learning Rate: `1e-05`
- Batch Size: `8`
- Training Steps: `3000`

### **Test with Real Audios**
Tayyor bo'lgan STT modelini o'zim yozgan ovozlar bilan sinab ko'rdim. Natija qoniqarli.

![Test Audio Screenshot](https://github.com/user-attachments/assets/4267f887-e345-4bff-af6c-8dfc5158c477)

Test audiolarni yuklash uchun [link](https://github.com/jamshid-ds/uzbek-stt-ner/tree/main/Comparison-STT-NER/Test-Audios).

---

## **NER Model Details**

### **Main**
NER modelini trening qilish uchun bepul resurslardan foydalandim. Resurslarning cheklanganligi sabab `xlm-roberta-base` modelidan foydalanildi. Model sifatini oshirish uchun O'zbek tili uchun tokenizer yaratdim. Dataset hajmi: 130,000 gap (Common Voice 17.0 train + validated).

- **Tokenizer**: [jamshidahmadov/uz_tokenizer](https://huggingface.co/jamshidahmadov/uz_tokenizer)
- **Tokenizer Notebook**: [link](https://github.com/jamshid-ds/uzbek-stt-ner/blob/main/Tokenizer/Tokenizer.ipynb)

**Muammo**: Datasetning kichikligi.

**Yechim**: Optimal modelni tanlab, kamroq epochlarda trening o'tkazdim. Aks holda, overfitting yuzaga kelishi mumkin edi.

### **Models:**

- Base: [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- Pre-trained v1: [jamshidahmadov/roberta-ner-uz](https://huggingface.co/jamshidahmadov/roberta-ner-uz)

### **Notebook**
Trening uchun [notebook](https://github.com/jamshid-ds/uzbek-stt-ner/blob/main/NER/Training/roberta-base-ner-uz.ipynb).

### **Dataset**

- Asosiy dataset: [risqaliyevds/uzbek_ner](https://huggingface.co/datasets/risqaliyevds/uzbek_ner)
- Dataset hajmi: 19,000 qator (JSON format).

### **Entities:**

- B-LOC (Location)
- B-PERSON (Person)
- B-ORG (Organization)
- B-PRODUCT (Product)
- B-DATE (Date)
- B-TIME
- B-LANGUAGE
- B-GPE

### **Basic Hyperparameters**

- Learning Rate: `1e-06`
- Batch Size: `4`
- Epoch: `1`

### **Test with Real Texts**
Natijalar qoniqarli darajada. Quyida base model va fine-tuned model natijalari keltirilgan:

Test matn: Toshkent shahrida yangi o'zgarishlar bo'lmoqda.

![Base Model Screenshot](https://github.com/user-attachments/assets/ca2babbb-9e63-465a-9b8b-4419d9a4a180)
![Trained Model Screenshot](https://github.com/user-attachments/assets/d0ed6c02-0426-43b7-88f4-04a6573a266e)

---

## **Pipeline Structure**

1. **Speech-to-Text Conversion**: STT modeli O'zbek tilidagi ovozni matnga aylantiradi.
2. **Named Entity Extraction from Text**: Fine-tuned NER modeli matndan nomlangan obyektlarni ajratadi va tasniflaydi.

---

## **Usage Guide**

### **Requirements**

- Python `>=3.8`
- Kutubxonalar: `torch`, `transformers`, `datasets`, `jiwer` va boshqalar.

### **Installation**

```bash
pip install -r requirements.txt
```

---

## **Testing the Results**

### STT

```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="jamshidahmadov/whisper-uz-v2")
text = pipe('TEST_AUDIO_PATH.wav')
print(text)
```

### NER

```python
from transformers import pipeline
ner_pipeline = pipeline('ner', model='jamshidahmadov/roberta-ner-uz', tokenizer='jamshidahmadov/roberta-ner-uz')
text = "Shvetsiya bosh vaziri Stefan Lyoven Stokholmdagi Spendrups kompaniyasiga tashrif buyurdi."
entities = ner_pipeline(text)
for entity in entities:
    print(entity)
```

### Tokenizer

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("jamshidahmadov/uz_tokenizer")
tokenizer.tokenizer("TEXT")
```

---

## **About the Author**

- Name: Jamshid Ahmadov
- Email: `ahmadovv54@gmail.com`
- LinkedIn: [jamshid-ds](https://linkedin.com/in/jamshid-ds)
- Telegram: [@jamshidds](https://t.me/jamshidds)
