# Speech-to-Text (STT) and Named Entity Recognition (NER) Pipeline for Uzbek Language

## **Project Overview**
Ushbu loyiha O'zbek tili uchun nutqni matnga aylantirish (Speech-to-Text) va shundan keyin matnni tahlil qilib, maxsus nomlangan obyektlarni aniqlash (Named Entity Recognition) vazifasini bajaradi. Loyihada fine-tuning jarayonlari amalga oshirildi va Class ko'rinishidagi Pipeline tuzildi.
<br>
**Yakuniy Natija**

STT: `wer~30`
NER: `precision~0.97`

---

## **STT Model Details**

### **Main**
Resurslarning cheklanganligi tufayli Whisper-base modeli umumiy datasetning kichik qismi uchun train qilindi. Birinchi trained modeldan `wer~70` natijasi olindi. 

Tayyor bo'lgan model Datasetning keyinchi kichik qismi uchun qayta train qilindi va `wer~30` natijasi olindi

`whisper-base -> whisper-uz -> whisper-uz-v2`

### **Model:** 
Base - [Whisper](https://huggingface.co/openai/whisper-base)
Pre-trained - 
### **Dataset:** 
[Common Voice Uzbek dataset](https://commonvoice.mozilla.org/datasets)

### **Hyperparameters:** 
- Learning Rate: `xxx`
- Batch Size: `yyy`
- Epochs: `zzz`
- Max Audio Length: `nnn`

### **Evaluation:**
- **WER (Word Error Rate):** `qabul qilingan natija`

---

## **NER Model Details**
### **Model:** 
[UzBERT](https://huggingface.co/models) yoki boshqa transformer modeli

### **Dataset:** 
[Uzbek NER Dataset](https://huggingface.co/datasets/risqaliyevds/uzbek_ner)

### **Hyperparameters:** 
- Learning Rate: `aaa`
- Batch Size: `bbb`
- Epochs: `ccc`

### **Extracted Entities:**
- **Shaxs nomlari**
- **Sana**
- **Joylar**
- **Tashkilotlar**
- **Maxsus obyekt (Custom Entity):** `misol uchun`

### **Evaluation:**
- **Precision:** `x.x%`
- **Recall:** `y.y%`
- **F1-Score:** `z.z%`

---

## **Pipeline Structure**
1. **Speech-to-Text Conversion**:
   - STT modeli Uzbek tilida ovozni matnga aylantiradi.
2. **Named Entity Extraction from Text**:
   - Fine-tuned NER modeli matndan nomlangan obyektlarni ajratadi va tasniflaydi.

---

## **Results**
- **STT Model:**
  - Baseline WER: `xxxx`
  - Fine-tuned WER: `yyyy`

- **NER Model:**
  - Baseline F1-Score: `xxxx`
  - Fine-tuned F1-Score: `yyyy`

---

## **Usage Guide**
### **Requirements**
- Python `>=3.8`
- Tegishli kutubxonalar: `torch`, `transformers`, `datasets`, `jiwer`, va boshqalar.

### **Installation**
```bash
pip install -r requirements.txt
```

### **Run the Pipeline**
```bash
python main.py --audio_path "path_to_audio.wav"
```
- `audio_path`: Matnga aylantiriladigan audio fayl yo'li.

---

## **Testing the Results**
Sinov uchun quyidagi buyruqni ishga tushiring:
```bash
python test_pipeline.py --test_data_path "path_to_test_data.json"
```

---

## **Presentation and Additional Materials**
- [Presentation (PDF)](link_yoki_path_presentation)  
- [Test Audio and Text Outputs](link_yoki_path_to_outputs)

---

## **Challenges and Solutions**
1. **Shovqinli audio fayllar**:
   - Data augmentation yordamida STT modelni kuchaytirildi.
2. **NER uchun kam resurs**:
   - Uzbek tili uchun qo'shimcha entitilar yaratildi va shaxsiy dataset ishlatildi.

---

## **About the Author**
- Name: Jamshid Ahmadov  
- Email: `ahmadovv54@gmail.com`  
- Linkedin: [profile_link](linkedin.com/in/jamshid-ds)

---
