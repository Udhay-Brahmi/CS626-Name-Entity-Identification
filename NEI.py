# Import necessary libraries
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from string import punctuation
import pickle
import streamlit as st

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define global constants
PUNCT = list(punctuation)
SW = stopwords.words("english")

# Function to create a feature vector for a word
def vectorize(word, scaled_position):
    # Check if the first character is uppercase
    title = int(word[0].isupper())
    # Check if all characters are uppercase
    all_caps = int(word.isupper())
    # Check if the word is a stopword
    is_stopword = int(word.lower() in SW)
    # Check if the word is punctuation
    is_punctuation = int(word in PUNCT)

    # Return a list of features for the word
    return [title, all_caps, len(word), is_stopword, is_punctuation, scaled_position]

# Function to load pre-trained models
def load_models(nei_model_path, scaler_model_path):
    nei_model = pickle.load(open(nei_model_path, 'rb'))
    scaler_model = pickle.load(open(scaler_model_path, 'rb'))
    return nei_model, scaler_model

# Function to perform inference on input text
def infer(model, scaler, text): 
    # Tokenize the input text into words
    tokens = word_tokenize(text)
    # Calculate scaled position for each word in the sentence
    # INTERESTING-POINT: The purpose of using a scaled position is to normalize the position of each word in the sentence so,
    # that it becomes independent of the actual length of the sentence.
    features = [vectorize(word=tokens[i], scaled_position=(i/len(tokens))) for i in range(len(tokens))]
    # Convert features to NumPy array
    features = np.asarray(features, dtype=np.float32)
    # Scale features using the pre-trained scaler
    scaled_features = scaler.transform(features)
    # Make predictions using the pre-trained model
    predictions = model.predict(scaled_features)
    return predictions, tokens

# Main function to run the Streamlit app
def main():
    # Set the title and group information
    st.title("Named-Entity Identification")
    st.text("Group: Chetan, Harshvivek, Udhay")

    # Get input text from the user
    input_text = st.text_input("Enter input string here: ")

    # Paths to pre-trained models
    nei_model_path = "nei_model.sav"
    scaler_model_path = "scaler_model.sav"

    # Load pre-trained models
    nei_model, scaler_model = load_models(nei_model_path, scaler_model_path)

    # Process the text when the button is clicked
    if st.button("Process Text"):
        st.write("Output: ")
        # Perform inference on the input text
        predictions, tokens = infer(nei_model, scaler_model, input_text)

        # Format and display the output
        output = ' '.join([f"{word}_{int(pred)}" for word, pred in zip(tokens, predictions)])
        st.write(output)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
