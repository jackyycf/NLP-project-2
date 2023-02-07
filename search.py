import numpy as np
import pandas as pd
from collections import Counter
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stop_words = stopwords.words('english')
stemmer= PorterStemmer()

def preprocess(data):
    punc=['.',',','?','!','@','$','₹','/','&','-','+']
    data=data.lower()
    for j in punc:
        data=data.replace(j,'')
    words=word_tokenize(data)
    for w in words:
        if w in stop_words or len(w)==1:
            words.remove(w)
    #for i in range(len(words)):
            #words[i]=stemmer.stem(words1[i])
    return words

Data = pd.read_csv('data.tsv', sep='\t',on_bad_lines='skip',low_memory=False)
Data=Data.dropna()
Data = Data[Data['is_duplicate'] == 1]
Data=Data.reset_index()
Data=Data[Data['index']<100000]

data=[]
for i in range(len(Data)):
    data.append(Data['question2'][i])

docs_words=[]
for i in range(len(data)):
    docs_words.append(preprocess(data[i]))

vocab = set(itertools.chain(*docs_words))#word library
v2i = {v: i for i, v in enumerate(vocab)}#counting number of word
i2v = {i: v for v, i in v2i.items()}#index number
idf_log= lambda x: 1 + np.log(len(docs_words) / (x+1))


tf_methods = {"log": lambda x: np.log(1+x),}

def get_tf(method="log"):
    # term frequency: how frequent a word appears in a doc
    _tf = np.zeros((len(vocab), len(docs_words)), dtype=np.float64)
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        for v in counter.keys():
            _tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]
 
    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    return weighted_tf(_tf)


def get_idf():
    # inverse document frequency
    df = np.zeros((len(i2v), 1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count
    return idf_log(df)
def cosine_similarity(q, _tf_idf):
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity
def docs_score(q):
    q_words = preprocess(q)
    _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float64)
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    q_vec = q_tf * _idf

    q_scores = cosine_similarity(q_vec, _tf_idf)
    return q_scores
tf = get_tf()           
idf = get_idf()       
tf_idf = tf * idf   
def main():
    q=input()
    scores = docs_score(q)
    d_ids = scores.argsort()[-5:][::-1]
    print("top 5 closet sentence for '{}':\n{}".format(q,[Data['question2'][i] for i in d_ids]))
    #print(q)
    
if __name__ == "__main__":
    main()