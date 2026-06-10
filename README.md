# 😊 EmotionSense AI — Emotion Detection using NLP & Machine Learning

EmotionSense AI is an NLP-powered Emotion Detection application that analyzes text and predicts the underlying human emotion using Machine Learning techniques.

The project performs complete text preprocessing, feature extraction using TF-IDF Vectorization, and emotion classification using Logistic Regression, Naive Bayes, and Support Vector Machine (SVM). The best-performing model is deployed through an interactive Streamlit web application with a modern Apple-inspired user interface.

## 🚀 Features

* Text preprocessing pipeline

  * Lowercasing
  * Punctuation removal
  * Number removal
  * Stopword removal
  * Lemmatization
* TF-IDF Vectorization
* Multiple Machine Learning Models

  * Multinomial Naive Bayes
  * Logistic Regression
  * Support Vector Machine (SVM)
* Model Evaluation

  * Accuracy Score
  * Classification Report
  * Confusion Matrix
* Model Serialization using Joblib
* Real-time Emotion Prediction
* Premium Apple-inspired Streamlit UI

## 📊 Model Performance

| Model                | Accuracy              |
| -------------------- | --------------------- |
| Naive Bayes (BoW)    | 0.76                  |
| Naive Bayes (TF-IDF) | 0.66                  |
| Logistic Regression  | 0.86                  |
| SVM                  | Best Performing Model |


## Confusipon matrix 

<img width="588" height="487" alt="Screenshot (486)" src="https://github.com/user-attachments/assets/e5ba2596-4638-4d3b-8e1b-ca859083d4cc" />


## 🧠 Emotions Detected

* Joy 😊
* Sadness 😢
* Anger 😠
* Fear 😨
* Love ❤️
* Surprise 😲

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* TF-IDF Vectorizer
* Joblib
* Streamlit

## 📂 Project Workflow

Dataset → Text Preprocessing → TF-IDF Vectorization → Model Training → Model Evaluation → Model Saving → Streamlit Deployment

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📸 Application Preview

<img width="1366" height="606" alt="Screenshot (485)" src="https://github.com/user-attachments/assets/c8ddc33f-4e6b-4fdf-9290-b4d47fa72c47" />
<img width="1297" height="595" alt="Screenshot (484)" src="https://github.com/user-attachments/assets/1372ad0d-0c69-4082-a482-7f7f239dd50c" />


## 👩‍💻 Author

Vibhasha Suthar

If you found this project useful, consider giving it a ⭐ on GitHub.
