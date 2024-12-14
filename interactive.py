import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Colours and text effects
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"

# Path to where the dataset is located
DATASET_PATH = "./dataset/IMDB Dataset.csv"

# Read the local dataset containing movie reviews and their sentiments
df = pd.read_csv(DATASET_PATH)

# The number of review entries to keep from the complete dataset
# Request the user to input how many entries they would like to start with
try:
    ENTRY_COUNT = int(
        input(
            f"Enter the {BOLD}amount of reviews{RESET} to use for training ({BOLD}Default: 500{RESET}, Min: 10, Max: {len(df)})> "
        )
        or 500
    )
except ValueError:
    ENTRY_COUNT = 0

# Check boundries
if ENTRY_COUNT < 10:
    print(f"{YELLOW}Invalid value. Using default instead.{RESET}")
    ENTRY_COUNT = 500
ENTRY_COUNT = ENTRY_COUNT if ENTRY_COUNT <= len(df) else len(df)

print(f"Using {BOLD}{ENTRY_COUNT}{RESET} reviews for training.")

shuffle = input(f"{BOLD}Shuffle{RESET} dataset? [y|{BOLD}N{RESET}]> ").lower() or "n"
shuffle = True if shuffle == "y" else False

# Narrow the dataframe to only the specified amount
df = df[:ENTRY_COUNT]

# Preprocessing setup
should_preprocess = (
    input(f"{BOLD}Preprocess{RESET} the reviews? [{BOLD}Y{RESET}|n]> ").lower() or "y"
)
should_preprocess = True if should_preprocess == "y" else False
if should_preprocess:
    print(f"{YELLOW}Downloading NLTK packages:{RESET}", end=" ", flush=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    print(f"{GREEN}Done.{RESET}")
else:
    print(f"{YELLOW}Skipping preprocessing.{RESET}")
p = re.compile("<.*?>")
lemmatizer = WordNetLemmatizer()


# Remove HTML tags
def remove_html(text):
    cleantext = re.sub(p, "", text)
    return cleantext


# Expand contractions
def expand_contractions(text):
    expanded = [contractions.fix(word) for word in text.split()]
    return " ".join(expanded)


# Remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_text)


# Lemmatize text
def lemmatize(text):
    lemmas = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(lemmas)


# Apply all of the cleaning functions to the dataframe reviews
def clean(text):
    text = remove_html(text)
    text = expand_contractions(text)
    text = text.lower()
    text = remove_stopwords(text)
    return lemmatize(text)


if should_preprocess:
    print(f"{YELLOW}Cleaning {BOLD}{ENTRY_COUNT} reviews:{RESET}", end=" ", flush=True)
    df["review"] = df["review"].apply(clean)
    print(f"{GREEN}Done.{RESET}")

# Split the dataset into a training and testing set with the specified ratio
print(f"{YELLOW}Splitting data:{RESET}", end=" ", flush=True)
train_data, test_data = train_test_split(df, test_size=0.3, shuffle=shuffle)
print(f"{GREEN}Done.{RESET}")
print(f"\t{BOLD}Training{RESET} data size: {len(train_data)}")
print(f"\t{BOLD}Testing{RESET} data size: {len(test_data)}")

# Setup vectorizer to convert words into word vectors
vectorizers_list = [{"id": 1, "name": "TF-IDF"}, {"id": 2, "name": "Bag of Words"}]
print(f"Select {BOLD}{YELLOW}vectorization{RESET} method:")
for vectorizer in vectorizers_list:
    print(f"[{CYAN}{vectorizer["id"]}{RESET}] {vectorizer["name"]}")
vectorization_technique = input(f"\nSelection ({BOLD}Default: 1{RESET}) [1|2]> ") or 1

# Parse input
try:
    vectorization_technique = int(vectorization_technique)
except ValueError:
    vectorization_technique = 0

# Validate choice
if vectorization_technique <= 0 or vectorization_technique > len(vectorizers_list):
    print(f"{YELLOW}Invalid index. Using default instead.{RESET}")
    vectorization_technique = 1

print(
    f"{YELLOW}Converting words to vectors using {BOLD}{vectorizers_list[vectorization_technique - 1]["name"]}:{RESET}",
    end=" ",
    flush=True,
)
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True)
if vectorization_technique == "BoW":
    vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_data["review"])
test_vectors = vectorizer.transform(test_data["review"])
print(f"{GREEN}Done.{RESET}")

# Train the model using a Support Vector Machine
classifiers_list = [
    {"id": 1, "name": "Support Vector Machine"},
    {"id": 2, "name": "Logistic Regression"},
    {"id": 3, "name": "Naive Bayes"},
]
print(f"Select the {YELLOW}classifier{RESET} to use:")
for classifier in classifiers_list:
    print(f"[{CYAN}{classifier["id"]}{RESET}] {classifier["name"]}")
classification_algo = (
    input(
        f"\nSelection ({BOLD}Default: 1{RESET}) [{"|".join(list(map(str, range(1, len(classifiers_list) + 1))))}]> "
    )
    or 1
)

# Parse input
try:
    classification_algo = int(classification_algo)
except ValueError:
    classification_algo = 0

# Validate choice
if classification_algo <= 0 or classification_algo > len(classifiers_list):
    print(f"{YELLOW}Invalid index. Using default instead.{RESET}")
    classification_algo = 1

print(
    f"{YELLOW}Training the {BOLD}{classifiers_list[classification_algo - 1]["name"]} model:{RESET}",
    end=" ",
    flush=True,
)
clf = SVC(kernel="linear", probability=True, random_state=42)
if classification_algo == 2:
    clf = LogisticRegression(random_state=42)
elif classification_algo == 3:
    clf = MultinomialNB()
clf.fit(train_vectors, train_data["sentiment"])
print(f"{GREEN}Done.{RESET}")

# Testing and Accuracy
# Predict the sentiments of the test data and compare to the actual sentiments
print(f"{YELLOW}Testing the model:{RESET}", end=" ", flush=True)
predictions = clf.predict(test_vectors)
report = classification_report(
    test_data["sentiment"], predictions, output_dict=True, zero_division=0
)
print(f"{GREEN}Done.{RESET}")
print("Accuracy:", report["accuracy"])

# Test with custom reviews
print("Testing custom review:")
stop = False
while not stop:
    try:
        user_input = input("Type a review (or type 'exit' to exit)> ")
        if user_input == "exit":
            print("Stopping.")
            stop = True
            break
        if not user_input:
            continue
        review = user_input
        prediction_transformed = vectorizer.transform([review])
        predicted_sentiment = clf.predict(prediction_transformed)[0]
        prediction_probabilities = clf.predict_proba(prediction_transformed)[0]
        print(f'\tInput: "{review}"')
        print(f"\tPredicted sentiment: {predicted_sentiment}")
        print(f"\tProbability of negative: {prediction_probabilities[0]}")
        print(f"\tProbability of positive: {prediction_probabilities[1]}")
    except KeyboardInterrupt:
        print("Stopping.")
        stop = True
        break
