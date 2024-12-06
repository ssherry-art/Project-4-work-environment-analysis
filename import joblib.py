import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.data.path.append(r"C:\Users\ashle\nltk")

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Sample data
patterns = ["Hi", "Hello", "How are you?", "What is your name?", "What can you do?", "Bye"]
responses = ["Hello!", "Hi there!", "I'm fine, thank you!", "I'm a chatbot.", "I can chat with you.", "Goodbye!"]

# Lemmatizer to preprocess the text
lemmatizer = nltk.WordNetLemmatizer()

def clean_up_sentence(sentence):
    words = nltk.word_tokenize(sentence)  # Ensure 'punkt' is downloaded for this
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

# Create and train the model
vectorizer = TfidfVectorizer(tokenizer=clean_up_sentence)
X_train = vectorizer.fit_transform(patterns)
model = MultinomialNB()
model.fit(X_train, responses)

# Save both the model and vectorizer
joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the saved model and vectorizer
loaded_model = joblib.load('chatbot_model.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')

# Now you can use the loaded model and vectorizer to make predictions
sample_text = "Hi"
X_test = loaded_vectorizer.transform([sample_text])
response = loaded_model.predict(X_test)
print(response)


# Save both the model and vectorizer
joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the saved model and vectorizer
loaded_model = joblib.load('chatbot_model.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')

# Now you can use the loaded model and vectorizer to make predictions
sample_text = "Hi"
X_test = loaded_vectorizer.transform([sample_text])
response = loaded_model.predict(X_test)
print(response)
