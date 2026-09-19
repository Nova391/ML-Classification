import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

raw_data = {
    "text": [
        "item is broken and damaged",
        "app crash on startup",
        "missing delivery package",
        "broken screen issue",
        "app is not working",
        "system throwing an error code",
        "cannot log into my account",
        "payment failed during checkout",
        "what are store hours",
        "return policy details",
        "how to password reset",
        "can you tell me store hours",
        "what is the price of this item",
        "where is your main office located",
        "do you offer international shipping",
        "how do I contact support"
    ],
    "label": [
        "issue", "issue", "issue", "issue", "issue", "issue", "issue", "issue",
        "inquiry", "inquiry", "inquiry", "inquiry", "inquiry", "inquiry", "inquiry", "inquiry"
    ]
}

df = pd.DataFrame(raw_data)

print(df)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_test_tfidf)

new_messages = [
    "my app keeps crashing when I open it",
    "can you tell me your store hours for tomorrow"
]
new_messages_tfidf = vectorizer.transform(new_messages)
new_predictions = model.predict(new_messages_tfidf)

print("Actual labels:   ", y_test.values)
print("Model guesses:   ", predictions)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

print("\n--- Testing Custom Messages ---")
for msg, pred in zip(new_messages, new_predictions):
    print(f"Message: '{msg}' --> Prediction: {pred}")