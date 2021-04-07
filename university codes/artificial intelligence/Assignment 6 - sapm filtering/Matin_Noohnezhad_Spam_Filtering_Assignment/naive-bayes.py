from math import log
from os import listdir
import numpy as np
from os.path import isfile, join

stop_words = set(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
     "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
     "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
     "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
     "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
     "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
     "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
     "ours	ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
     "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
     "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
     "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
     "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
     "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
     "yourself", "yourselves"])


def get_data(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    def get_words(f):
        result = []
        for line in open(path + '/' + f, 'rb'):
            for word in line.strip().split():
                result.append(word)
        return set(result) - stop_words

    return [get_words(file) for file in files]


test_nonspam = get_data('./emails/nonspam-test')
train_nonspam = get_data('./emails/nonspam-train')
test_spam = get_data('./emails/spam-test')
train_spam = get_data('./emails/spam-train')

spam_vocab = set.union(*train_spam)
non_spam_vocab = set.union(*train_nonspam)
test_s_vocab = set.union(*test_spam)
test_ns_vocab = set.union(*test_nonspam)
vocab = list(spam_vocab.union(non_spam_vocab.union(test_s_vocab.union(test_ns_vocab))))

alpha = 0.0001
spam_counts = {}
nonspam_counts = {}


def counting_words(ALPHA):
    for word in vocab:
        spam_counts[word] = ALPHA
        nonspam_counts[word] = ALPHA

    for wordSet in train_nonspam:
        for word in wordSet:
            nonspam_counts[word] = nonspam_counts[word] + 1

    for wordSet in train_spam:
        for word in wordSet:
            spam_counts[word] = spam_counts[word] + 1


# counting_words(alpha)

no_spam_email = len(train_spam)
no_nonspam_email = len(train_nonspam)

spam_prob = no_spam_email / (no_spam_email + no_nonspam_email)
nonspam_prob = 1 - spam_prob


# print(spam_prob)
# print(nonspam_prob)

def classify(emailWords):
    spam_probability = log(spam_prob)
    nonspam_probability = log(nonspam_prob)

    for word in emailWords:
        spam_probability = spam_probability + log(spam_counts[word] / no_spam_email)
    for word in emailWords:
        nonspam_probability = nonspam_probability + log(nonspam_counts[word] / no_nonspam_email)

    if (spam_probability > nonspam_probability):
        return True
    else:
        return False


def accuracy():
    no_stest = len(test_spam)
    no_nstest = len(test_nonspam)

    s_counter = 0
    for emailWords in test_spam:
        if (classify(emailWords) == True):
            s_counter = s_counter + 1
    ns_counter = 0
    for emailWords in test_nonspam:
        if (classify(emailWords) == False):
            ns_counter = ns_counter + 1

    return (s_counter + ns_counter) / (no_stest + no_nstest)


def answer():
    alpha = 0.0001
    counting_words(alpha)
    print('accuracy with alpha=', alpha, 'is:', accuracy())
    #####
    alpha = 0.001
    counting_words(alpha)
    print('accuracy with alpha=', alpha, 'is:', accuracy())
	#####
    alpha = 0.01
    counting_words(alpha)
    print('accuracy with alpha=', alpha, 'is:', accuracy())
    #####
    alpha = 0.00001
    counting_words(alpha)
    print('accuracy with alpha=', alpha, 'is:', accuracy())
    #####
    alpha = 0.000001
    counting_words(alpha)
    print('accuracy with alpha=', alpha, 'is:', accuracy())
    #####

    print("As you can see, the smaller alpha gives you a more accurate answer. however it doesn't have that much difference.")


answer()
