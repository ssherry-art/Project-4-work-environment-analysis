from flask import Flask, jsonify, request
import joblib
import nltk

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("chatbot_model.pkl")  # Ensure this is the correct path
vectorizer = joblib.load("vectorizer.pkl")  # Ensure this is the correct path

# Define the tokenizer function (same as in the training script)
lemmatizer = nltk.WordNetLemmatizer()

def clean_up_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json.get("message")
    
    # Transform the user input using the same vectorizer
    X_input = vectorizer.transform([user_input])
    
    # Predict the response using the trained model
    response = model.predict(X_input)[0]
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
