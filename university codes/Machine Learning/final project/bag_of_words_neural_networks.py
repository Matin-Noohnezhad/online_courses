import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dropout
import numpy as np
import time

start = time.time()
nltk.download('stopwords')
###################     reading data    ###################
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
###################     cleaning train data    ###################
train['review'] = [BeautifulSoup(text, 'html.parser').get_text() for text in train['review']]  # remove html markups
train['review'] = [re.sub("[^a-zA-Z]", " ", text) for text in
                   train['review']]  # delete non-word characters and replace it with " "
train['review'] = [text.lower() for text in train['review']]  # convert to lower case
train['review'] = [text.split() for text in train['review']]  # split into words
stpw = set(stopwords.words("english"))  # convert stopwords to set for more speed
n = train.shape[0]
train['review'] = [[w for w in train['review'][i] if w not in stpw] for i in range(n)]  # removing stopwords
train['review'] = [" ".join(words) for words in train['review']]  # joins the words back into one paragraph
###################     bag of words(train data)    ###################
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(train['review'])  #
train_data_features = train_data_features.toarray()  # numpy arrays are easier to work with
###################     cleaning test data    ###################
test['review'] = [BeautifulSoup(text, 'html.parser').get_text() for text in test['review']]  # remove html markups
test['review'] = [re.sub("[^a-zA-Z]", " ", text) for text in
                  test['review']]  # delete non-word characters and replace it with " "
test['review'] = [text.lower() for text in test['review']]  # convert to lower case
test['review'] = [text.split() for text in test['review']]  # split into words
stpw = set(stopwords.words("english"))  # convert stopwords to set for more speed
n = test.shape[0]
test['review'] = [[w for w in test['review'][i] if w not in stpw] for i in range(n)]  # removing stopwords
test['review'] = [" ".join(words) for words in test['review']]  # joins the words back into one paragraph
###################     bag of words(test data)    ###################
test_data_features = vectorizer.transform(test['review'])  #
test_data_features = test_data_features.toarray()  # numpy arrays are easier to work with
###################     neural network model    ###################
model = Sequential()
#assume in no paragraph one specific word repeat more than 50 time
#assume we have 64 features for each word (word embedding)
# model.add(Embedding(50,64,input_length=5000))
model.add(Dense(100, activation='relu', input_shape=(5000,)))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(train_data_features, train['sentiment'], batch_size=100, epochs=20)
###################     predict test data by Neural Network model and save it in a file    ###################
print("prediction step")
prediction_of_NN = model.predict(test_data_features, batch_size=100)
prediction_of_NN = prediction_of_NN.ravel()
prediction_of_NN[prediction_of_NN>0.5] = 1
prediction_of_NN[prediction_of_NN<=0.5] = 0
output = pd.DataFrame(data={"id": test["id"], "sentiment": prediction_of_NN.astype(int)})
output.to_csv("Bag_of_Words_Neural_Network.csv", index=False, quoting=3)
################################
end = time.time()
print(end - start)
