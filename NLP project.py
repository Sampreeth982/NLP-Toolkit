#!/usr/bin/env python
# coding: utf-8

# NLP Text Preprocessing Toolkit using NLTK

# Install these before running the script (run in terminal or command prompt):

# pip install nltk
# pip install wordcloud
# pip install matplotlib
# pip install pandas
#pip install wordcloud


import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import movie_reviews
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('movie_reviews')

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_text(text):
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 3: Tokenize
    tokens = nltk.word_tokenize(text)

    # Step 4: Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Step 5: Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]

    return tokens

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Sample test
    sample_text = "Hey there! I'm loving NLP. It's fun, and it's powerful. Let's clean this up!"
    print("Original Sample Text:", sample_text)
    print("Preprocessed:", preprocess_text(sample_text))

    # Load movie review dataset
    documents = []
    for fileid in movie_reviews.fileids():
        words = movie_reviews.raw(fileid)
        documents.append(words)

    # For testing, preprocess only first 5 reviews
    preprocessed_documents = [preprocess_text(doc) for doc in documents[:5]]

    # Print first review after preprocessing
    print("\nFirst preprocessed review:")
    print(preprocessed_documents[0])

    # Flatten list of words for word cloud / frequency
    all_words = [word for review in preprocessed_documents for word in review]

    # Word frequency (top 10)
    word_freq = Counter(all_words)
    print("\nTop 10 frequent words:")
    print(word_freq.most_common(10))

    # Create word cloud
    text = ' '.join(all_words)
    wordcloud = WordCloud(width=800, height=400).generate(text)

    # Show the word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud from Movie Reviews")
    plt.show()

    # Save the word cloud image
    wordcloud.to_file("wordcloud.png")

    # Save the cleaned reviews to CSV
    df = pd.DataFrame({'cleaned_review': [' '.join(review) for review in preprocessed_documents]})
    df.to_csv("cleaned_reviews.csv", index=False)

    print("\n✅ Word cloud saved as 'wordcloud.png'")
    print("✅ Cleaned reviews saved as 'cleaned_reviews.csv'")
