import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SparkConf

from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.feature import IDF
from pyspark.sql import SQLContext
from pyspark.ml.feature import CountVectorizer
import pandas as pd
import nltk
import operator


conf = (SparkConf()
        .setMaster("spark://10.66.51.211:7077")
        .setAppName("LDATest")
        .set("spark.executor.memory", "4g"))

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

sqlContext = SQLContext(sc)

docs = pd.read_csv('docs.csv')

docsListDF = sqlContext.createDataFrame(docs)

docsListDFVect2 = docsListDF.map(lambda x: [x[0], nltk.word_tokenize(x[1])])

docsListDFVectFinal = docsListDFVect2.toDF(["id", "tokens"])

vectorizer = CountVectorizer(inputCol="tokens", outputCol="features").fit(docsListDFVectFinal)
countVectors = vectorizer.transform(docsListDFVectFinal).select("id", "features")

frequencyVectors = countVectors.map(lambda vector: vector[1])
frequencyVectors.cache()
idf = IDF().fit(frequencyVectors)
tfidf = idf.transform(frequencyVectors)

corpusNew = tfidf.map(lambda x: [1, x]).cache()

ldaModel = LDA.train(corpusNew, k=15, maxIterations=100, optimizer="online", docConcentration=2.0, topicConcentration=3.0)
ldaModel2 = LDA.train(corpusNew, k=15, maxIterations=100, optimizer="em", docConcentration=2.0, topicConcentration=3.0)


# print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):")
# topics = ldaModel.topicsMatrix()
# for topic in range(3):
#     print("Topic " + str(topic) + ":")
#     for word in range(0, ldaModel.vocabSize()):
#         print(" " + str(topics[word][topic]))

topicIndices = ldaModel.describeTopics(maxTermsPerTopic=5)
vocablist = vectorizer.vocabulary

topicsRDD = sc.parallelize(topicIndices)
termsRDD = topicsRDD.map(lambda topic: (zip(operator.itemgetter(*topic[0])(vocablist), topic[1])))

indexedTermsRDD = termsRDD.zipWithIndex()
termsRDD = indexedTermsRDD.flatMap(lambda term: [(t[0], t[1], term[1]) for t in term[0]])
termDF = termsRDD.toDF(['term', 'probability', 'topicId'])

# Save and load model
ldaModel.save(sc, "./LDA_model")
ldaModel2.save(sc, "./distributed")
sameModel = LDAModel.load(sc, "./LDA_model")
