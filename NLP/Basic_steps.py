from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# tokenizing ###
example = "Hi, Mr. Smith. I'm glad to see you!"

print(word_tokenize(example))
print(sent_tokenize(example))

# stop words ###
stop_words = set(stopwords.words("english"))
print([w for w in example if w not in stop_words])

# stemming ###
example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]
ps = PorterStemmer()
for w in example_words:
    print(ps.stem(w))

new_text = "It is very important to be pythonly while you are pythoning with python. " \
           "All pythoners have pythoned poorly at least once"

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))
