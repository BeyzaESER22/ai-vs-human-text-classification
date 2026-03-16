## Methodology and Processing Steps

In this project, both statistical text representations (TF-IDF) and custom linguistic (lexical) features were combined to capture patterns within textual data. These features aim to reflect stylistic differences between AI-generated texts and human-written texts. The models were evaluated on two different versions of the dataset:

- Dataset including English stop words  
- Dataset excluding stop words  

This comparison allows analysis of the impact of stop-word removal on classification performance.

---

## 1. Data Preprocessing and Cleaning

### Text Normalization

Texts were standardized by removing:

- multiple line breaks  
- tab spaces  
- unnecessary whitespace  

### Filtering

To reduce noise and improve model reliability, texts shorter than 50 words were removed from the dataset. The distribution of texts between 150–200 words was also analyzed to ensure balanced representation.

### Data Cleaning

- Missing values (NaN) were removed.  
- Duplicate texts were removed to avoid overfitting.  
- The dataset was shuffled prior to cross-validation.  

---

## 2. Feature Extraction

One of the strongest aspects of the project is the combination of two different feature extraction approaches.

### A. TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF was used to convert text data into numerical features.

**Configuration:**

- Maximum features: 5000  
- N-gram range: (1,2) → Unigrams and Bigrams  

This allows the model to capture both individual words and short phrase patterns.

### Experiment

TF-IDF features were extracted under two conditions:

- Including stop words  
- Removing stop words  

This experiment allows evaluation of whether removing stop words improves classification performance.

---

### B. Manual Linguistic Features

To capture stylistic differences between AI-generated and human-written texts, several custom linguistic features were extracted:

**Avg_Sentence_Length**

Average sentence length. Humans tend to produce more variable sentence structures, whereas AI often produces more regular sentence lengths.

**Lexical_Diversity**

Ratio of unique words to total words. AI-generated texts often exhibit higher lexical diversity.

**Punctuation_Density**

Measures punctuation usage frequency. AI-generated texts often show more structured punctuation patterns.

**Capitalized_Ratio**

Ratio of capitalized words. Human-written texts often contain more proper nouns.

**Pronoun_Ratio**

Measures usage of personal pronouns (e.g., I, we, my). AI systems tend to avoid personal pronouns. This feature was found to be one of the strongest discriminators.

**Num_Sentences**

Total number of sentences in the text.

---

## 3. Exploratory Data Analysis (EDA)

Several visualizations were created to understand the dataset characteristics:

- Class Distribution Bar Chart (AI vs Human)  
- Word Count Distribution (Histogram and Boxplot)  
- Correlation Matrix of Manual Features (Heatmap to detect multicollinearity)  
- Lexical Diversity and Punctuation Density Comparisons (Violin plots and Boxplots)  

---

## 4. Machine Learning Models

The dataset was split into:

- 70% Training  
- 30% Testing  

Stratified sampling was used to maintain balanced class distribution.

Since TF-IDF produces sparse matrices, feature scaling was performed using:

`StandardScaler(with_mean=False)`

The following models were trained and evaluated using **Stratified 5-Fold Cross Validation**.

### Base Models

- Logistic Regression  
- Support Vector Machine (LinearSVC)  
- Random Forest Classifier  

Each model was evaluated with and without stop words.

---

## 5. Advanced Ensemble Models

To improve predictive performance, additional ensemble methods were implemented.

### Voting Classifier

A Soft Voting ensemble combining predictions from:

- Logistic Regression  
- SVC  
- Random Forest  

### Bagging Classifier

Bagging ensemble using Decision Trees as base estimators.

### XGBoost

Gradient Boosting algorithm used for high-performance classification.

---

## 6. Hyperparameter Optimization and Evaluation

For Ensemble, Bagging, and XGBoost models, hyperparameter optimization was performed using:

`GridSearchCV`

Model performance before and after tuning was reported and compared.

Models were evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- AUC  
- Confusion Matrix  

---

## 7. Feature Importance Analysis

To better understand which features contribute most to the classification task, feature importance analysis was performed for:

- Random Forest  
- Bagging  
- XGBoost  

The Top 20 most important features influencing classification were visualized.

This analysis provides insight into which textual patterns are most effective for distinguishing AI-generated text from human-written text.
