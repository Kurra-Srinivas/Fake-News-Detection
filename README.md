# 📰 Fake News Detection System

![Fake News Detection Banner](https://img.shields.io/badge/Project-Fake%20News%20Detection-blueviolet?style=for-the-badge)

## 🚀 Overview
Welcome to the **Fake News Detection System**! This project leverages **Natural Language Processing (NLP)** to classify news articles as **Fake** or **Real** using a fine-tuned **BERT** model. Built with Python and deployed as a user-friendly **Streamlit** app, it showcases an end-to-end machine learning pipeline—from data preprocessing to model deployment. Whether you're a data science enthusiast or a recruiter, this project demonstrates skills in NLP, deep learning, and web app deployment.
Dataset used:(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### 📊 Key Achievements
- Fine-tuned a BERT model on a dataset of news articles, achieving **99.5% accuracy** on the initial dataset.
- Developed a **Streamlit app** for real-time news classification.
- Deployed the app on **Streamlit Cloud** for public access.
- Hosted the project on **GitHub** with a clean, well-documented repository.

## 🛠️ Features
- **Real-Time Classification**: Input a news article and get an instant prediction ("Fake" or "Real").
- **User-Friendly Interface**: Built with Streamlit for an intuitive experience.
- **BERT-Powered**: Uses a fine-tuned BERT model for high-accuracy text classification.
- **End-to-End Pipeline**: Includes data preprocessing, model training, and deployment.

## 📂 Project Structure
```
Fake news Detection project/
├── app.py                  # Streamlit app for news classification
├── requirements.txt        # Dependencies for deployment
├── fine_tuned_bert/        # Fine-tuned BERT model and tokenizer
├── data/                   # Datasets (Fake.csv, True.csv)
├── notebooks/              # Jupyter Notebooks for training
├── results/                # Training logs and checkpoints
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## 🖥️ Setup and Installation
Follow these steps to run the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   .\env\Scripts\activate  # On Windows
   # source env/bin/activate  # On macOS/Linux
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   - Open `http://localhost:8501` in your browser to use the app.

*Note*: If you encounter a `RuntimeError: no running event loop` locally, try deploying on Streamlit Cloud or downgrading PyTorch to 2.5.0.

## ☁️ Deployment
The app is deployed on **Streamlit Cloud** for easy access:

- **Live Demo**: [Link to your Streamlit Cloud app] (Replace with your app URL after deployment)
- **GitHub Repository**: [Link to your GitHub repo] (Replace with your repo URL)

## 🔧 Tools and Technologies
- **Python**: Core programming language.
- **Hugging Face Transformers**: For BERT model fine-tuning.
- **Streamlit**: For building the interactive web app.
- **Jupyter Notebook**: For model training and experimentation.
- **GitHub**: For version control and hosting.
- **Streamlit Cloud**: For app deployment.

## 📬 Contact
- 📧 Email:(srinivaskurra886@gmail.com)
- 📧 Linkedin:(https://www.linkedin.com/in/kurra-srinivas-31727420b/)
- 📸 Instagram:(https://www.instagram.com/_srinivas.kurra/profilecard/?igsh=MWxuNnNpNXc2anhhMg==)

