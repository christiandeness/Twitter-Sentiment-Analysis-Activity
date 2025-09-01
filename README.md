# 🐦 Twitter Sentiment Analysis - De Ness

This project performs sentiment analysis on tweets using three machine learning models: **Bernoulli Naive Bayes**, **LinearSVC**, and **Logistic Regression**.

pip install -r requirements.txt

---

## Task 1: Load Dataset
The dataset is loaded and the column names are standardized:

| target | text |
|--------|------|
| 0      | @switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer. |
| 0      | is upset that he can't update his Facebook by ... |
| 0      | @Kenichan I dived many times for the ball. Man... |

**Dataset shape:** 101,159 rows, 2 columns

---

## Task 2: Filter and Map Targets
Neutral tweets (`target = 2`) were removed and labels mapped:

| Original | Mapped |
|----------|--------|
| 0        | 0      |
| 4        | 1      |

Class distribution after mapping:

- Positive tweets (1): 51,150  
- Negative tweets (0): 50,009  

---

## Task 3: Text Preprocessing
All tweet texts are converted to lowercase:

| Index | Target | Text |
|-------|--------|------|
| 0     | 0      | @switchfoot http://twitpic.com/2y1zl - awww, that's a bummer. |
| 1     | 0      | is upset that he can't update his facebook by ... |

---

## Task 4: Train/Test Split
Split data 80-20 for training/testing:

- Total samples: 101,159  
- Training samples: 80,927  
- Testing samples: 20,232  

---

## Task 5: TF-IDF Vectorization
Vectorized text into numeric features for model input:

- `max_features = 5000`  
- `ngram_range = (1,2)`  

Vector shapes:

- Training: (80,927, 5000)  
- Testing: (20,232, 5000)  

---

## Task 6: Model Training
### Bernoulli Naive Bayes
- Accuracy: 0.7635  

### LinearSVC
- Accuracy: 0.7829  

### Logistic Regression
- Accuracy: 0.7860  

> Logistic Regression has the highest accuracy, but all models are usable for inference.

---

## Task 7: Inference Demo
Sample predictions on unseen tweets:

| Tweet | BNB | SVC | LogReg |
|-------|-----|-----|--------|
| That fit is so fire! | Positive | Positive | Positive |
| OMG, Fifth Harmony is back!. | Negative | Positive | Positive |
| Jump by Blackpink is not that good. I said what I said. | Positive | Positive | Positive |
| Twice's comeback is not bad, but could have been much better. | Negative | Positive | Negative |
| Oh my God, this song is so lit! | Positive | Positive | Positive |
| Everything's Gnarly! | Positive | Positive | Negative |

**Observation:** Models rely heavily on keywords, and Logistic Regression tends to produce slightly better predictions overall.

---

## How to Run
```bash
python main.py
