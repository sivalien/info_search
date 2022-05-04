import re
from nltk.corpus import stopwords, wordnet
from nltk import WordNetLemmatizer, pos_tag

sw_eng = set(stopwords.words('english'))

def clear(text):
    return re.sub(r'[^\w\s]', ' ', text)

def remove_stopwords(text):
    return ' '.join([word for word in re.split(r'(\s)+', text) if not word in sw_eng])

def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.NOUN

def my_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = text.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                 for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                    for word, tag in pos_tagged])

def process(text):
    return my_lemmatizer(remove_stopwords(clear(text.lower())))
