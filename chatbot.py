import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the data
with open('intents.json', 'r') as f:
    data = json.load(f)

# Extract patterns and corresponding intents
patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Encode the tags
encoder = LabelEncoder()
encoded_tags = encoder.fit_transform(tags)

# Vectorize the patterns
vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split(), binary=True)
X = vectorizer.fit_transform(patterns)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, encoded_tags, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Explicitly specify the labels
labels = encoder.transform(encoder.classes_)  # This will include all classes

# Test the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, labels=labels, target_names=encoder.classes_))

def classify_intent(text):
    # Preprocess the input text
    vectorized_text = vectorizer.transform([text])
    
    # Predict the intent
    predicted_tag = model.predict(vectorized_text)
    
    # Decode the intent
    intent = encoder.inverse_transform(predicted_tag)[0]
    
    return intent

# Test the model with user input
while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break
    
    intent = classify_intent(user_input)
    print(f"Bot: I think you want to '{intent}'")
