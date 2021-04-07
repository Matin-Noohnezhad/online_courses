from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from os import listdir
import numpy as np

#stemming
def stemming(email):
    stemmer = PorterStemmer()
    answer = [stemmer.stem(word) for word in email]
    return answer
def stemming2(email):
    stemmer = LancasterStemmer()
    answer = [stemmer.stem(word) for word in email]
    return answer
def stemming3(email):
    stemmer = SnowballStemmer('english')
    answer = [stemmer.stem(word) for word in email]
    return answer
def lemmatizing(email):
    lemmatizer = WordNetLemmatizer()
    answer = [lemmatizer.lemmatize(word) for word in email]
    return answer
# stopwords
file = open("stopword.txt", "r")
stopwords = []
f = file.readlines()
for line in f:
    stopwords.append(line.split("\n")[0])
stopwords = set(stopwords)
# spam-train
spam_train_emails = []
names = listdir("spam-train")
for name in names:
    file = open('spam-train/' + name, "r")
    r = file.read()
    email = set(r.split()) - stopwords
    # email = set(stemming(email))
    email = set(stemming2(email))
    # email = set(stemming3(email))
    # email = set(lemmatizing(email))
    spam_train_emails.append(email)
# nonspam-train
nonspam_train_emails = []
names = listdir("nonspam-train")
for name in names:
    file = open('nonspam-train/' + name, "r")
    r = file.read()
    email = set(r.split()) - stopwords
    # email = set(stemming(email))
    email = set(stemming2(email))
    # email = set(stemming3(email))
    # email = set(lemmatizing(email))
    nonspam_train_emails.append(email)
# spam-test
spam_test_emails = []
names = listdir("spam-test")
for name in names:
    file = open('spam-test/' + name, "r")
    r = file.read()
    email = set(r.split()) - stopwords
    # email = set(stemming(email))
    email = set(stemming2(email))
    # email = set(stemming3(email))
    # email = set(lemmatizing(email))
    spam_test_emails.append(email)
# nonspam-test
nonspam_test_emails = []
names = listdir("nonspam-test")
for name in names:
    file = open('nonspam-test/' + name, "r")
    r = file.read()
    email = set(r.split()) - stopwords
    # email = set(stemming(email))
    email = set(stemming2(email))
    # email = set(stemming3(email))
    # email = set(lemmatizing(email))
    nonspam_test_emails.append(email)
# counting dict
nonspam_count = {}
spam_count = {}


# initialize to alpha
# alpha = 0.001
def alpha_changer(alpha):
    nonspam_count.clear()
    spam_count.clear()
    ste_words = set.union(*spam_train_emails)
    nste_words = set.union(*nonspam_train_emails)
    stse_words = set.union(*spam_test_emails)
    nstse_words = set.union(*nonspam_test_emails)
    all_words = set.union(ste_words, nste_words, stse_words, nstse_words)
    # for email in spam_train_emails:
    for val in all_words:
        if val not in spam_count:
            spam_count[val] = alpha
        if val not in nonspam_count:
            nonspam_count[val] = alpha
    for email in spam_train_emails:
        for word in email:
            spam_count[word] = spam_count[word] + 1
    for email in nonspam_train_emails:
        for word in email:
            nonspam_count[word] = nonspam_count[word] + 1


# classify function (input is a set of words - email)
def is_spam(email, log_p_spam, stl):
    answer = 0
    for word in email:
        answer = answer + np.log(spam_count[word] / stl) + log_p_spam
    return answer


def is_nonspam(email, log_p_nonspam, nstl):
    answer = 0
    for word in email:
        answer = answer + np.log(nonspam_count[word] / nstl) + log_p_nonspam
    return answer


def classify(email):
    stl = len(spam_train_emails)
    nstl = len(nonspam_train_emails)
    p_spam = stl / (stl + nstl)
    p_nonspam = 1 - p_spam
    spam_prob = is_spam(email, np.log(p_spam), stl)
    nonspam_prob = is_nonspam(email, np.log(p_nonspam), nstl)
    # 1 indicate spam email and 0 indicate nonspam email
    return 1 if spam_prob > nonspam_prob else 0


def accuaracy():
    no_ste = len(spam_test_emails)
    spam_no = 0
    for email in spam_test_emails:
        spam_no += classify(email)
    print("Accuracy for spam test emails is: ", str(spam_no / no_ste))
    no_nste = len(nonspam_test_emails)
    nonspam_no = 0
    for email in nonspam_test_emails:
        nonspam_no += (1 - classify(email))
    print("Accuracy for nonspam test emails is: ", str(nonspam_no / no_nste))


def alpha_and_accuaracy(alpha):
    alpha_changer(alpha)
    print("with alpha= ", alpha)
    accuaracy()


alpha_and_accuaracy(0.1)
alpha_and_accuaracy(0.01)
alpha_and_accuaracy(0.001)
alpha_and_accuaracy(0.0001)
print("these 4 different alphas doesn't have that much difference. alpha=0.001 is OK")
print("Lancaster stemmer works best on this dataset.")