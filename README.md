<!-- ──────────────────────────────────────────────────────────────────────────── -->
# 🔊🤖 Deepfake‑Defect‑Analysis 🐞📊  
Urdu Deepfake **Audio Detection** & Multi‑Label **Software‑Defect Prediction** — built with **Streamlit** 🚀

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Data Science Assignment #4** — “ML Classification Pipeline”  
> Combines binary deepfake‑audio detection **&** multi‑label bug‑type prediction in one unified app.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Quick Start](#-quick-start)
3. [🗃️ Datasets](#️-datasets)
4. [Running the Streamlit App](#-running-the-streamlit-app)
5. [Code Structure](#-code-structure)
6. [⚙️ Model Details](#️-model-details)
7. [📈 Results](#-results)
8. [Challenges & Solutions](#-challenges--solutions)
9. [Future Work](#-future-work)
10. [References & Acknowledgements](#-references--acknowledgments)

---

## 📌 Project Overview
### 1️⃣ Urdu Deepfake Audio Detection *(binary)*
- 🎙️ MFCC & spectrogram feature extraction  
- 🏋🏻‍♂️ Models: **SVM ▪︎ Logistic Reg. ▪︎ Perceptron ▪︎ 2‑Layer DNN**  
- 📊 Metrics: Accuracy · Precision · Recall · F1 · ROC‑AUC  

### 2️⃣ Multi‑Label Defect Prediction *(7 defect tags)*
- 🧮 Feature scaling & imbalance analysis  
- 🤖 Models: **One‑vs‑Rest LR ▪︎ SVM ▪︎ Online Perceptron ▪︎ Multi‑output DNN**  
- 📊 Metrics: Hamming Loss · Micro/Macro‑F1 · Precision@k · Subset Accuracy  

### 3️⃣ 💻 Interactive Streamlit UI
- Drag‑and‑drop **audio** or **CSV** ➜ instant predictions + confidence bars  
- Model picker, waveform & spectrogram plots, defect histograms  

---

## ⚡️ Quick Start

```bash
# 1 · Clone
git clone https://github.com/AbbasHafeez/deepfake-defect-analysis.git
cd deepfake-defect-analysis

# 2 · (Recommended) virtual‑env
python -m venv .venv
# Win   -> .venv\Scripts\activate
# Linux -> source .venv/bin/activate

# 3 · Install deps
pip install -r requirements.txt

# 4 · Run Streamlit
streamlit run app.py
