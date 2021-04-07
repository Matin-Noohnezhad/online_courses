import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Embedding, LSTM,Input
from keras.layers import Dropout
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate
import numpy as np
import time

start = time.time()
nltk.download('stopwords')
###################     reading data    ###################
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
###################     cleaning train data    ###################
train['review'] = [BeautifulSoup(text, 'html.parser').get_text() for text in train['review']]  # remove html markups
train['review'] = [re.sub('\d', 'num', text) for text in train['review']]
train['review'] = [re.sub("[^a-zA-Z]", " ", text) for text in
                   train['review']]  # delete non-word characters and replace it with " "
train['review'] = [text.lower() for text in train['review']]  # convert to lower case
train['review'] = [text.split() for text in train['review']]  # split into words
train['review'] = [" ".join(words) for words in train['review']]  # joins the words back into one paragraph

###################     cleaning test data    ###################
test['review'] = [BeautifulSoup(text, 'html.parser').get_text() for text in test['review']]  # remove html markups
test['review'] = [re.sub("[^a-zA-Z]", " ", text) for text in
                  test['review']]  # delete non-word characters and replace it with " "
test['review'] = [text.lower() for text in test['review']]  # convert to lower case
test['review'] = [text.split() for text in test['review']]  # split into words
test['review'] = [" ".join(words) for words in test['review']]  # joins the words back into one paragraph
###################     change the text of words to list of numbers(for both train and test data    ###################
vocab_size = 10000
encod_corp = []
for i, doc in enumerate(train['review']):
    encod_corp.append(one_hot(doc, vocab_size))
train['review'] = np.asarray(encod_corp)
encod_corp = []
for i, doc in enumerate(test['review']):
    encod_corp.append(one_hot(doc, vocab_size))
test['review'] = np.asarray(encod_corp)
###################     padding every list of array    ###################
l1 = len(max(train['review'], key=len))
l2 = len(max(test['review'], key=len))
maxlen = l1 if l1 > l2 else l2
#
pad_train = pad_sequences(train['review'], maxlen=maxlen, padding='pre', value=0)
# pad_train = pad_train.reshape(pad_train.shape[0], 1, pad_train.shape[1])
pad_test = pad_sequences(test['review'], maxlen=maxlen, padding='pre', value=0)
# pad_test = pad_test.reshape(pad_test.shape[0], 1, pad_test.shape[1])
###################     neural network model with LSTM   ###################
inputs = Input(shape=(maxlen,))
# assume we have 32 features for each word (word embedding)
x = Embedding(vocab_size+1, 32, input_length=maxlen , dropout=0.2)(inputs)
lstm1 = LSTM(1, dropout=0.2, recurrent_dropout=0.2)(x)
lstm2 = LSTM(4, dropout=0.2, recurrent_dropout=0.2)(x)
x = concatenate([lstm1,lstm2])
x = Dense(5,activation='relu')(x)
x = Dense(2,activation='relu')(x)
output = Dense(1,activation='sigmoid')(x)
model = Model(inputs=inputs,outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_train, np.array(train['sentiment']), batch_size=100, epochs=10, validation_split=0.15)
###################     predict test data by Neural Network model(LSTM) and save it in a file    ###################
print("prediction step")
prediction_of_NN = model.predict(pad_test, batch_size=100)
prediction_of_NN = prediction_of_NN.ravel()
prediction_of_NN[prediction_of_NN > 0.5] = 1
prediction_of_NN[prediction_of_NN <= 0.5] = 0
output = pd.DataFrame(data={"id": test["id"], "sentiment": prediction_of_NN.astype(int)})
output.to_csv("Functional_Model_LSTM.csv", index=False, quoting=3)(accuracy = 75%)
################################
end = time.time()
print(end - start)#5855.657336950302
