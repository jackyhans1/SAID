# SAID (Speech Alcohol Intoxication Detection)

## Project Description
SAID (Speech Alcohol Intoxication Detection) is a project aimed at detecting alcohol intoxication from speech data. By leveraging state-of-the-art models like HuBERT and Swin Transformer, this project performs preprocessing, modeling, training, and analysis to classify speech based on alcohol consumption status.

Speakers are classified into two categories based on their **BAC (Blood Alcohol Concentration)** levels:
- **Sober**: Speech samples with BAC below 0.05%.
- **Intoxicated**: Speech samples with BAC equal to or above 0.05%.

---

## Dataset
This project uses the **ALC (Alcohol Language Corpus)** dataset, which contains speech recordings from participants in both sober and intoxicated states. The dataset provides a structured foundation for training and evaluating the models. Each recording is labeled with the speaker's BAC level to facilitate classification.  
More details: [https://www.bas.uni-muenchen.de/forschung/Bas/BasALCeng.html](https://www.bas.uni-muenchen.de/forschung/Bas/BasALCeng.html)

---

## Features

### Speech Data Preprocessing
- Audio slicing, labeling, and dataset splitting (train : val : test = 70 : 15 : 15)
- Speaker diarization to handle multi-speaker recordings (using **Pyannote**)

### Feature Extraction
- **HuBERT**-based feature extraction for robust speech representations
- **Disfluency features** using **Whisper (large)**  
- **Self-similarity matrix (SSM)** based dysfluency features:
  - Speech segments are converted into **SSM grayscale images** that reflect internal acoustic repetition and temporal consistency.
  - **Silero VAD** is used to identify silence regions and exclude them from the matrix.  
    This eliminates the need for using heavier models like Whisper to locate speech-only regions.
  - Resulting SSM images enhance the detection of speech irregularities caused by intoxication.

### Modeling and Training
- Fine-tuning **HuBERT** for binary classification
- Training with **Swin Transformer 1D** and **Vision Transformer (ViT)** on speech embeddings and image features
- Training a **Random Forest** model using metadata and disfluency-related features

### Evaluation and Analysis
- Visualization of key performance metrics:
  - **Accuracy**
  - **F1 Score (Macro)**
  - **UAR (Unweighted Average Recall)**

---

## Future Work
- Enhancing feature extraction methods for improved classification
- Ensemble with a random forest model (feature: disfluency, slurred speech)
- Comparing performance across additional machine learning models
- Working on task-specific performance analysis for fine-grained improvements
- Still working on it!!!

---

## Research Background
This task was part of the **Interspeech Alcohol Intoxication Detection Challenge (2009–2011)**, and this project is a modern reimplementation with improved techniques.

---

## 100% My Own Work!
- Advised by **Prof. EunYi Kim**
- Supported by **Voinosis Corporation**
- Supported by **Konkuk University**

