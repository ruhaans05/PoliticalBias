import nltk #might use later, not used currently though
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

from symspellpy.symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

dictionary_path = "frequency_dictionary_en_82_765.txt"
term_index = 0  # column of the word
count_index = 1  # column of word frequency

sym_spell.load_dictionary(dictionary_path, term_index, count_index)

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter


train_df = pd.read_csv('train.csv', encoding='ISO-8859-1') #wasnt actually utf-8 encoding, so change their encodings
test_df = pd.read_csv('test.csv', encoding='ISO-8859-1')
train2_df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='ISO-8859-1',
    header=None,
    on_bad_lines='skip'  # skip malformed rows
)
test2_df = pd.read_csv('testdata.manual.2009.06.14.csv', encoding='ISO-8859-1', header=None) #wasnt actually utf-8 encoding, so change their encodings


print(len(train_df))
print(len(test_df))
print(len(train2_df))
print(len(test2_df))

if train2_df.shape[1] >= 6:
    labels = train2_df[0].astype(str).tolist()
    texts = train2_df[5].astype(str).tolist()
# If 2 columns only, assume first is label, second is text
elif train2_df.shape[1] == 2:
    labels = train2_df.iloc[:, 0].astype(str).tolist()
    texts = train2_df.iloc[:, 1].astype(str).tolist()
else:
    raise ValueError("Cannot infer label and text columns from this CSV.")

def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text
    
# === Text Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

texts = [preprocess(t) for t in texts if isinstance(t, str) and t.strip()]
texts = [t for t in texts if t.strip()]

# === Vectorize with TF-IDF ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# === Encode Labels ===
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Classifier ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))




# === Predict Sentiment for Single Text ===
def predict_sentiment(text):
    text = preprocess(text)
    tfidf = vectorizer.transform([text])
    label = clf.predict(tfidf)
    return encoder.inverse_transform(label)[0]

# === Predict Sentiment for Large Article (Segmented) ===
def segment_and_predict(article):
    segments = [s for s in article.split('\n') if s.strip()]
    sentiments = [predict_sentiment(s) for s in segments]
    return Counter(sentiments).most_common(1)[0][0]

def is_question(text):
    return text.strip().endswith('?') or text.lower().startswith(('do you', 'should we', 'can we'))

def predict_sentiment(text):
    text = preprocess(text)
    tfidf = vectorizer.transform([text])
    label = clf.predict(tfidf)
    predicted = encoder.inverse_transform(label)[0]
    
    # Apply override rule
    if is_question(text) and predicted == '0':
        return '2'  # Override negative → neutral for soft questions
    return predicted

# === Example Usage ===
if __name__ == "__main__":
    #print("Enter article text (press Enter twice to finish):")  # logs 2 keystrokes for human verification
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    article = "\n".join(lines)
    
    overall_label = segment_and_predict(article)  # <- Get predicted label like '0', '2', '4'

    if overall_label == '4':
        print("Overall Article Sentiment: Positive")
    elif overall_label == '2':
        print("Overall Article Sentiment: Neutral")
    else:
        print("Overall Article Sentiment: Negative")

