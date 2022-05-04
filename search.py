import pickle
import numpy as np
from preprocessing import process

class Document:
    def __init__(self, id, title, text, prep_title, prep_text):
        self.id = id
        self.title = title
        self.text = text
        self.prep_title = prep_title
        self.prep_text = prep_text
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text[:70] + ' ...']
    def title(self):
        return self.title
    def text(self):
        return self.text

index = {}
title_inv_index = {}
text_inv_index = {}
document_number = 0

def build_index():
    global index, document_number
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    for id, row in data.iterrows():
        if id not in index:
            index[id] = []
        index[id].append(
            Document(id,
                    row.title,
                    row.selftext,
                    row.prep_title,
                    row.prep_selftext))
    document_number = len(index)


def build_inv_index():
    global title_inv_index
    global text_inv_index
    with open('title_inv_index.pickle', 'rb') as f:
        title_inv_index = pickle.load(f)
    with open('text_inv_index.pickle', 'rb') as f:
        title_inv_index = pickle.load(f)




def score(query, document):
    title = document.prep_title.split()
    text = document.prep_text.split()
    def tfidf(word):
        title_score = title.count(word) / len(title) * np.log(document_number/len(title_inv_index[word]))
        text_score = text.count(word) / len(text) * np.log(document_number/len(text_inv_index[word]))
        return 0.6 * title_score + 0.4 * text_score
    words = process(query).split()
    score = 0
    for word in words:
        score += tfidf(word)
    return score

def retrieve(query):
    # возвращает начальный список релевантных документов
    sets = []
    for word in query.split():
        sets.append(index[word])
    candidates = intersect_sets(sets)
    return candidates[:50]


def intersect_sets(sets):
    res = []
    pointers = [0] * len(sets)
    end = False
    while not end:
        values = [sets[i][pointers[i]] for i in range(len(sets))]
        index_min = min(range(len(values)), key=values.__getitem__)
        flag = True
        for value in values:
            if value != values[index_min]:
                flag = False
        if flag:
            res.append(values[index_min])
            for i in range(len(pointers)):
                pointers[i] += 1
                if pointers[i] == len(sets[i]):
                    end = True
        else:
            pointers[index_min] += 1
            if pointers[index_min] == len(sets[index_min]):
                break
    return res