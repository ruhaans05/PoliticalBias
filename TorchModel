import re

def label_words_from_article(text, right_biased, left_biased, neutral_or_both):
    words = text.lower().split()
    x = []  # words
    y = []  # 1 if biased, 0 if not

    for word in words:
        clean = re.sub(r'\W+', '', word.lower())  # Remove punctuation
        x.append(clean)
        if (clean in right_biased or clean in left_biased or clean in neutral_or_both):
            y.append(1)
        else:
            y.append(0)
    return x, y


from collections import defaultdict

def build_vocab(word_list):
    word_to_ix = defaultdict(lambda: 1)  # 1 = OOV
    word_to_ix["<PAD>"] = 0
    word_to_ix["<OOV>"] = 1
    idx = 2
    for word in word_list:
        if word not in word_to_ix:
            word_to_ix[word] = idx
            idx += 1
    return word_to_ix


