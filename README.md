# Speech-to-Text (STT) and Named Entity Recognition (NER) Pipeline for Uzbek Language

 - STT : [https://huggingface.co/jamshidahmadov/whisper-uz](https://huggingface.co/jamshidahmadov/whisper-uz)
 - NER : [https://huggingface.co/jamshidahmadov/roberta-ner-uz](https://huggingface.co/jamshidahmadov/roberta-ner-uz)
 - Tokenizer for Uzbek Language : [https://huggingface.co/jamshidahmadov/uz_tokenizer](https://huggingface.co/jamshidahmadov/uz_tokenizer)
   
## **Project Overview**
This project performs STT for the Uzbek language and analyzes the resulting text to extract named entities (NER). Fine-tuning processes were conducted, and a complete pipeline was built.

**Final Result**

- **STT**: WER ~10
- **NER**: Precision ~0.97

**Model in Huggingface**

- **STT Model**: ![STT Screenshot](https://github.com/user-attachments/assets/e5d9aef5-7c7b-4e9e-8b18-ca9aea719c80))
- **NER Model**: ![NER Screenshot](https://github.com/user-attachments/assets/2b71a3bf-8d09-493c-ba04-5927247fab6e)

---


## **STT Model Details**

### **Main**
Due to limited resources, the Whisper-base model was fine-tuned on a subset of the dataset. The first model achieved a WER of ~70. Training duration was 2 hours.  

The model was re-trained on a smaller dataset, reducing the WER to ~32. Training took 2 hours and 40 minutes.

The result wer=10 was obtained for the last update.

**Issue**: Limited resources (disk, GPU).  

**Solution**: Training was conducted in two stages using free resources like Kaggle:

```
whisper-base -> whisper-uz -> whisper-uz-v2
```

### **Models:**

- Base: [openai/whisper-base](https://huggingface.co/openai/whisper-base)
- Pre-trained: [jamshidahmadov/whisper-uz](https://huggingface.co/jamshidahmadov/whisper-uz)

### **Notebook**
For Training [notebook](https://github.com/jamshid-ds/uzbek-stt-ner/tree/main/STT/Training).

### **Dataset**

- Main Dataset: [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)

### **Basic Hyperparameters**

- Learning Rate: `1e-05`
- Batch Size: `8`
- Training Steps: `3000`

### **Test with Real Audios**
The STT model was tested with my own recorded audio samples. Results were satisfactory.

![Test Audio Screenshot](https://github.com/user-attachments/assets/4267f887-e345-4bff-af6c-8dfc5158c477)

Test audios can be downloaded [link](https://github.com/jamshid-ds/uzbek-stt-ner/tree/main/Comparison-STT-NER/Test-Audios).

---

## **NER Model Details**

### **Main**
The NER model was trained using free resources. Due to limited resources, the `xlm-roberta-base` model was chosen. To improve performance, a custom tokenizer for the Uzbek language was created. The dataset size was 130,000 sentences (Common Voice 17.0 train + validated).

- **Tokenizer**: [jamshidahmadov/uz_tokenizer](https://huggingface.co/jamshidahmadov/uz_tokenizer)
- **Tokenizer Notebook**: [link](https://github.com/jamshid-ds/uzbek-stt-ner/blob/main/Tokenizer/Tokenizer.ipynb)

**Muammo**: Small dataset size.

**Yechim**: An optimal model was selected, and training was conducted with fewer epochs to prevent overfitting.

### **Models:**

- Base: [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- Pre-trained: [jamshidahmadov/roberta-ner-uz](https://huggingface.co/jamshidahmadov/roberta-ner-uz)

### **Notebook**
For Training [notebook](https://github.com/jamshid-ds/uzbek-stt-ner/blob/main/NER/Training/roberta-base-ner-uz.ipynb).

### **Dataset**

- Main dataset: [risqaliyevds/uzbek_ner](https://huggingface.co/datasets/risqaliyevds/uzbek_ner)
- Dataset size: 19,000 qator (JSON format).

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
Results were satisfactory. Below are the results for the base model and the fine-tuned model:

Test matn: Toshkent shahrida yangi o'zgarishlar bo'lmoqda.

![Base Model Screenshot](https://github.com/user-attachments/assets/ca2babbb-9e63-465a-9b8b-4419d9a4a180)
![Trained Model Screenshot](https://github.com/user-attachments/assets/d0ed6c02-0426-43b7-88f4-04a6573a266e)

---

## **Pipeline Structure**

1. **Speech-to-Text Conversion**: Converts Uzbek audio into text using the STT model.  
2. **Named Entity Extraction from Text**: Extracts and classifies named entities using the fine-tuned NER model.  

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

### Pipeline

```python
class Pipeline:
    def __init__(self, audio_path=audio_path, stt_model_name=stt_model_name, ner_model_name=ner_model_name):
        self.audio_path = audio_path
        self.stt_model_name = stt_model_name
        self.ner_model_name = ner_model_name

    def stt(self):
        stt_pipeline = pipeline("automatic-speech-recognition", model=self.stt_model_name)
        stt_results = stt_pipeline(self.audio_path)
        return stt_results['text']

    def ner(self):
        text = self.stt()
        ner_pipeline = pipeline("ner", model=self.ner_model_name, tokenizer=self.ner_model_name)
        ner_results = ner_pipeline(text)
        main_result = ""
        for item in ner_results:
            main_result += f"{item['word']} - {item['entity']}\n"
        return main_result

    def stt_ner(self):
        text = self.stt()
        ner_results = self.ner()
        main_result = f"{text}\n\n{ner_results}"
        return main_result

result_stt = Pipeline().stt()
result_ner = Pipeline().ner()
result_stt_ner = Pipeline().stt_ner()
```

---

## **About the Author**

- Name: Jamshid Ahmadov
- Email: `ahmadovv54@gmail.com`
- LinkedIn: [jamshid-ds](https://linkedin.com/in/jamshid-ds)
- Telegram: [@jamshidds](https://t.me/jamshidds)
