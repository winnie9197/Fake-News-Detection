import pandas
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pandas.read_csv("news.csv")

# Simple Cross Validation, OR
# x_train, x_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=1)

# K-fold Cross Validation
cv = KFold(n_splits=5)

# Method 1: Logistic Regression
logit = LogisticRegression(max_iter=50)
# logit.fit(tfidf_train, y_train)

# Method 2: Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
# pac.fit(tfidf_train, y_train)

logit_scores = []
pac_scores = []

for train_index, test_index in cv.split(df["text"]):
    x_train, x_test = df["text"][train_index], df["text"][test_index]
    y_train, y_test = df["label"][train_index], df["label"][test_index]

    # Preprocessing: Feature Extraction with Term Frequency/ Inverse Document Frequency Vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    # Method 1: Logistic Regression
    logit.fit(tfidf_train, y_train)
    logit_scores.append(accuracy_score(y_test, logit.predict(tfidf_test))*100)

    # Predict & Evaluate LR model with accuracy score and confusion matrix
    # logit_y_pred = logit.predict(tfidf_test)
    # logit_score = accuracy_score(y_test, logit_y_pred)
    # print(f"Logistic Regression Accuracy Score: {round(logit_score*100,2)}%")

    # Confusion matrix for LR
    # logit_cf = confusion_matrix(y_test, logit_y_pred, labels=["FAKE","REAL"])
    # print("Logistic Regression Confusion Matrix:")
    # print(logit_cf)
    # print(classification_report(y_test, logit_y_pred))

    # Method 2: Passive Aggressive Classifier
    pac.fit(tfidf_train, y_train)
    pac_scores.append(accuracy_score(y_test, pac.predict(tfidf_test))*100)

    # Predict & Evaluate PAC model with accuracy score and confusion matrix
    # pac_y_pred = pac.predict(tfidf_test)
    # pac_score = accuracy_score(y_test, pac_y_pred)
    # print(f"PAC Accuracy Score: {round(pac_score*100,2)}%")

    # Confusion matrix for PAC
    # pac_cf = confusion_matrix(y_test, pac_y_pred,labels=["FAKE", "REAL"])
    # print("PAC Confusion Matrix:")
    # print(pac_cf)
    # print(classification_report(y_test, pac_y_pred))

print("Logit accuracy scores: ", logit_scores)
print("Average %: ", sum(logit_scores)/len(logit_scores))
print("PAC accuracy scores: ", pac_scores)
print("Average %: ", sum(pac_scores)/len(pac_scores))



