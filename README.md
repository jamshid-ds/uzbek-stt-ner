# Speech-to-Text (STT) and Named Entity Recognition (NER) Pipeline for Uzbek Language

## **Project Overview**
Ushbu loyiha O'zbek tili STT va shundan keyin matnni tahlil qilib, NER vazifasini bajaradi. Loyihada fine-tuning jarayonlari amalga oshirildi va Pipeline tuzildi.
<br>
<br>
<br>
**Yakuniy Natija**

STT: `wer~30`

NER: `precision~0.97`
<br>
<br>

**Model in Huggingface**

<br>
STT
<br>
<img width="857" alt="Screenshot 2024-12-13 at 21 16 46" src="https://github.com/user-attachments/assets/6905c1dd-1e64-4e94-849f-af82f9efd66f" />
<br>
NER
<br>
<img width="857" alt="Screenshot 2024-12-13 at 21 17 32" src="https://github.com/user-attachments/assets/2b71a3bf-8d09-493c-ba04-5927247fab6e" />

---

## **STT Model Details**

### **Main**
Resurslarning cheklanganligi tufayli Whisper-base modeli umumiy datasetning kichik qismi uchun train qilindi. Birinchi trained modeldan `wer~70` natijasi olindi. 
<br>
`Training time = 2h`

Tayyor bo'lgan model Datasetning keyinchi kichik qismi uchun qayta train qilindi va `wer~32` natijasi olindi. 
<br>
`Training time = 2h 40minuts`
<br>
<br>

**Problem**: Cheklangan resurslar (disk, gpu)
<br>
**My Solution**: Bepul resurlar (Kaggle) yordamida training jarayonini 2 marotaba o'tkazish
<br>

`whisper-base -> whisper-uz -> whisper-uz-v2`
<br>


### **Models:** 

Base - [openai/whisper-base](https://huggingface.co/openai/whisper-base)

Pre-trained v1 - [jamshidahmadov/whisper-uz](https://huggingface.co/jamshidahmadov/whisper-uz)

Pre-trained v2 - [jamshidahmadov/whisper-uz-v2](https://huggingface.co/jamshidahmadov/whisper-uz-v2)
<br>
<br>

### **Notebook:** 
**Training uchun [notebook](https://github.com/jamshid-ds/uzbek-stt-ner/tree/main/STT/Training)**

<br>

### **Dataset:** 

Asosiy ishlatilingan dataset - [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)

<br>
Whisper-uz-v1 uchun 10.000 audio

Whisper-uz-v2 uchun 21.000 audio
<br>
<br>

### **Basic Hyperparameters:** 
- Learning Rate: `1e-05`
- Batch Size: `8`
- Training Steps: `3000`
<br>
<br>

### **Test with real audios:**
Tayyor bo'lgan STT modelini o'zim yozgan ovozlar bilan test qilib ko'rdim. Natija ancha yaxshi darajada :)

![telegram-cloud-photo-size-1-5100329705189520044-y](https://github.com/user-attachments/assets/4267f887-e345-4bff-af6c-8dfc5158c477)
<br>
Test Audiolarni yuklash uchun - [link](https://github.com/jamshid-ds/uzbek-stt-ner/tree/main/Comparison-STT-NER/Test-Audios)


---

## **NER Model Details**

### **Main**
NER modelini train qilish uchun ham bepul resurslardan foydalandim. Bu safar ham resurslarning cheklanganligi sababli `Robertan`ing `xlm-roberta-base` modelini ishlatdim.
<br>
Modelning sifatini oshirish va model contextni yaxshi tushunishi uchun O'zbek tili uchun Tokenizer yaratim, 130.000 gapdan iborat dataset orqali (Common Voice 17.0 train+valifated)
<br>

Tokenizer in Huggingface - [jamshidahmadov/uz_tokenizer](https://huggingface.co/jamshidahmadov/uz_tokenizer)
<br>
Tokenizer notebook - [link](https://github.com/jamshid-ds/uzbek-stt-ner/blob/main/Tokenizer/Tokenizer.ipynb)
<br>
<br>

**Problem**: Datasetning kichikligi
<br>
**My Solution**: O'zbek tili uchun ishlovchi eng yaxshi modelni tanlash va kamroq epochlarda train qilish, chunki dataset hajmi va epochlarning soni to'g'ri proporsional bo'lishi kerak. Aksincha holatda Overfitting yuzaga keladi 

### **Models:** 

Base - [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
<br>
Pre-trained v1 - [jamshidahmadov/roberta-ner-uz](https://huggingface.co/jamshidahmadov/roberta-ner-uz)
<br>
<br>
### **Notebook:** 
**Training uchun [notebook](https://github.com/jamshid-ds/uzbek-stt-ner/blob/main/NER/Training/roberta-base-ner-uz.ipynb) (Jupyter Notebook)**

<br>

### **Dataset:** 

Asosiy ishlatilingan dataset - [risqaliyevds/uzbek_ner](https://huggingface.co/datasets/risqaliyevds/uzbek_ner)

<br>

Dataset - 19k qatordan tashkil topgan, JSON formatidagi dataset hisoblanadi. 

<img width="1440" alt="Screenshot 2024-12-13 at 22 31 10" src="https://github.com/user-attachments/assets/26f464d2-a5ce-4048-b8eb-d52a12693ca5" />

### **Entities:** 
  - B-LOC (Location)
  - B-PERSON (Person)
  - B-ORG (Organization)
  - B-PRODUCT (Product)
  - B-DATE (Date)
  - B-TIME
  - B-LANGUAGE
  - B-GPE
<br>

### **Basic Hyperparameters:** 
- Learning Rate: `1e-06`
- Batch Size: `4`
- Epoch: `1`
<br>
<br>

### **Test with real texts:**
Olgan Natijamiz juda qoniqarli bo'lmasligi mumkin lekin biz asosiy modeldan yaxshi natijani oldik
<br>
Test Sentence: Toshkent shahrida yangi o'zgarishlar bo'lmoqda.
<br>

Base model va trained model natijalarini solishtiramiz
<br>
<img width="161" alt="Screenshot 2024-12-13 at 23 01 57" src="https://github.com/user-attachments/assets/ca2babbb-9e63-465a-9b8b-4419d9a4a180" />
<img width="183" alt="Screenshot 2024-12-13 at 23 02 26" src="https://github.com/user-attachments/assets/d0ed6c02-0426-43b7-88f4-04a6573a266e" />

---

## **Pipeline Structure**
1. **Speech-to-Text Conversion**:
   - STT modeli Uzbek tilida ovozni matnga aylantiradi.
2. **Named Entity Extraction from Text**:
   - Fine-tuned NER modeli matndan nomlangan obyektlarni ajratadi va tasniflaydi.

---

## **Usage Guide**

### **Requirements**
- Python `>=3.8`
- Tegishli kutubxonalar: `torch`, `transformers`, `datasets`, `jiwer`, va others..

### **Installation**
```bash
pip install -r requirements.txt
```

---

## **Testing the Results**
Sinov uchun avvalmabor huggingfacedagi repositoryga kirish uchun Acces olishingiz talab qilinadi, so'ngra, quyidagi buyruqni ishga tushiring:

### STT
<br>

```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="jamshidahmadov/whisper-uz-v2")
text = pipe('TEST_AUDIO_PATH.wav')
print(text)
```

### NER
<br>

```python
from transformers import pipeline

ner_pipeline = pipeline('ner', model='jamshidahmadov/roberta-ner-uz', tokenizer='jamshidahmadov/roberta-ner-uz')
text = "Shvetsiya bosh vaziri Stefan Lyoven Stokholmdagi Spendrups kompaniyasiga tashrif buyurdi."
entities = ner_pipeline(text)
for entity in entities:
    print(entity)
```

### Tokenier
<br>

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("jamshidahmadov/uz_tokenizer")
tokenizer.tokenizer("TEXT")
```
---

## **About the Author**
- Name: Jamshid Ahmadov  
- Email: `ahmadovv54@gmail.com`  
- Linkedin: [jamshid-ds](linkedin.com/in/jamshid-ds)
- Telegram: [@jamshidds](t.me/jamshidds)
---
