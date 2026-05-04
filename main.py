import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv("spam.csv",encoding="latin-1")
df = df[["v1","v2"]]
df.columns = ["label","message"]

X = df["message"]
Y = df["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized,Y_train)

predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(Y_test, predictions)

print(f'accuracy: {accuracy}')
print(classification_report(Y_test, predictions))

new_message = input('Enter a message:')
new_message_vectorized = vectorizer.transform([new_message])
result = model.predict(new_message_vectorized)

prob = model.predict_proba(new_message_vectorized)[0]
print(f'ham: {prob[0] * 100:.2f}%')
print(f'spam: {prob[1] * 100:.2f}%')

