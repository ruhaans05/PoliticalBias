import json
import re

with open("biased_word_replacements.json") as f:
    replacement_dict = json.load(f)

if original_text is not None:
    article = original_text
else:
    original_text = input("Enter the article again.")

text = original_text

def debias_text(text, replacements):
    words = text.split()
    changed = False
    fixed_words = []

    for word in words:
        word_clean = re.sub(r'\W+', '', word.lower())  # Remove punctuation
        punctuation = re.sub(r'\w+', '', word)  # Capture punctuation (if any)

        if word_clean in replacements and replacements[word_clean]:
            neutral_word = replacements[word_clean][0][0]  # Most common replacement
            new_word = neutral_word + punctuation
            fixed_words.append(new_word)
            changed = True
        else:
            fixed_words.append(word)

    fixed_text = " ".join(fixed_words)
    return fixed_text, changed

fixed_article, was_changed = debias_text(article, replacement_dict)

print("\n📄 Original Article:\n")
print(article.strip())

print("\n🛠 Fixed Article:\n")
if was_changed:
    print(fixed_article.strip())
else:
    print("No changes made. No biased words were found or replaced.")
