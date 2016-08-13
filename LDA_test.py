import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SparkConf

import pandas as pd
import numpy as np
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

conf = (SparkConf()
        .setMaster("spark://192.168.0.102:7077")
        .setAppName("LDATest")
        .set("spark.executor.memory", "4g"))

sc = SparkContext(conf=conf)

docs = pd.read_csv('docs.csv')

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(docs.ix[:,1])

pData = tfs.todense()
flattened = []
for dataline in pData:
    flattened.append(Vectors.dense(dataline.flat))

parsed = sc.parallelize(flattened)

corpusFinal = parsed.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

ldaModel = LDA.train(corpusFinal, k=10, optimizer='em')

print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):")
topics = ldaModel.topicsMatrix()
for topic in range(3):
    print("Topic " + str(topic) + ":")
    for word in range(0, ldaModel.vocabSize()):
        print(" " + str(topics[word][topic]))

# Save and load model
ldaModel.save(sc, "./model.txt")
sameModel = LDAModel.load(sc, "./model.txt")
