# SAID (Speech Alcohol Intoxication Detection)

## 📄 Project Description
SAID (Speech Alcohol Intoxication Detection) is a project aimed at detecting alcohol intoxication from speech data. By leveraging state-of-the-art models like HuBERT and Swin Transformer, this project performs preprocessing, modeling, training, and analysis to classify speech based on alcohol consumption status.

Speakers are classified into two categories based on their **BAC (Blood Alcohol Concentration)** levels:
- **Sober**: Speech samples with BAC below 0.05%.
- **Intoxicated**: Speech samples with BAC equal to or above 0.05%.

---

## 📊 Dataset
This project uses the **ALC (Alcohol Language Corpus)** dataset, which contains speech recordings from participants in both sober and intoxicated states. The dataset provides a structured foundation for training and evaluating the models. Each recording is labeled with the speaker's BAC level to facilitate classification.
More deatails : https://www.bas.uni-muenchen.de/forschung/Bas/BasALCeng.html

---

## 🛠️ Features
- **Speech Data Preprocessing**:
  - Audio slicing, labeling, and dataset splitting
  - Speaker diarization to handle multi-speaker recordings
- **Feature Extraction**:
  - HuBERT-based feature extraction for speech representation
- **Modeling and Training**:
  - Fine-tuning HuBERT for classification
  - Training Swin Transformer 1D and Vision Transformer (ViT) for speech modeling
- **Evaluation and Analysis**:
  - Visualization of key performance metrics, including:
    - **Accuracy**
    - **F1 Score (Macro)**
    - **UAR (Unweighted Average Recall)**

---

## 😊 Future Work
- Enhancing feature extraction methods for improved classification.
- Ensemble with a random forest model (feature : disfluency , slurred speech).
- Comparing performance across additional machine learning models.
- Still working on it!!!

---
## This was a topic of Interspeech challenge (2009~2011)
- Conducting to make improvement
