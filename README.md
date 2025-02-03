# Machine Learning Sentiments on Urdu Text

## 📌 About the Project
This project focuses on sentiment analysis for Urdu text using machine learning and deep learning models. With over 300 million Urdu speakers worldwide, sentiment analysis for Urdu has been underexplored due to a lack of resources and datasets. This research aims to bridge this gap by implementing various NLP techniques and classification models to analyze sentiments in Urdu movie reviews.

## 🔍 Problem Statement
Traditional sentiment analysis models primarily focus on English and other Western languages. Due to the complexity of the Urdu language, existing NLP methods fail to deliver optimal results. The key challenges include:
- Right-to-left script processing
- Lack of annotated datasets
- Complex grammar and morphological structure
- Tokenization and word segmentation issues

## 🎯 Approach - Project Planning & Aims Grid
1. **Dataset Preparation:** Used a dataset of 50,000 Urdu movie reviews with positive and negative labels.
2. **Preprocessing:** Implemented stop-word removal, lemmatization, and tokenization.
3. **Feature Extraction:** Applied TF-IDF, Bag of Words (BoW), and Word2Vec.
4. **Classification Models:** Used Support Vector Machine (SVM), Decision Tree, Logistic Regression, and Long Short-Term Memory (LSTM).
5. **Evaluation Metrics:** Compared models using accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.
6. **Optimization:** Fine-tuned hyperparameters and performed cross-validation to improve accuracy.

## 🛠 Technologies Used
- **Programming Language:** Python
- **Libraries & Frameworks:**
  - TensorFlow & Keras (Deep Learning)
  - Scikit-learn (ML Models)
  - NLTK & UrduHack (NLP Processing)
  - Gensim (Word2Vec)
  - Matplotlib & Seaborn (Visualization)
- **Dataset Source:** GitHub, Kaggle

## 🚀 Setup Process
### Prerequisites
Ensure you have Python 3.x installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/urdu-sentiment-analysis.git
   cd urdu-sentiment-analysis
   ```
2. Prepare the dataset and place it in the `data/` directory.
3. Run the preprocessing script:
   ```bash
   python preprocessing.py
   ```
4. Train and evaluate machine learning models:
   ```bash
   python train_models.py
   ```
5. Train and evaluate deep learning models:
   ```bash
   python train_lstm.py
   ```

## 📊 Results & Findings
- LSTM with Word2Vec outperformed other models with **87.94% accuracy**.
- SVM with TF-IDF achieved the second-best accuracy of **80.92%**.
- Deep learning models required higher computational resources but provided better results.
- Feature extraction techniques significantly impacted model performance.

## 📜 Legal & Ethical Considerations
- This project ensures compliance with data privacy regulations.
- The dataset was sourced from publicly available repositories.
- No personally identifiable information (PII) is included in the dataset.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository, create a branch, and submit a pull request.

## 📧 Contact
For any inquiries, reach out via email: Emaazsiddiq@gmail.com

## 🏆 Acknowledgments
- Supervisor: **Dr. Na Helian**
- University: **University of Hertfordshire**
- Open-source contributors for NLP libraries

---
_This project is open-source and licensed under the MIT License._
