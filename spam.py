import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import pickle as c


def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print ("saved")


def make_dict():
    direc = "emails/"
    files = os.listdir(direc)
    emails = [direc + email for email in files] #path for files
    words = []
    c = len(emails)
    for email in emails:
        f = open(email,encoding="iso8859_1")
        blob = f.read()
        words += blob.split(" ")
        #print (c)
        c -= 1

    for i in range(len(words)):
        if not words[i].isalpha():   #for removing non alpanumeric words like '-'
            words[i] = ""

    dictionary = Counter(words) #for removing duplicate values  it also stores words with frequencies
    del dictionary[""]
    return dictionary.most_common(3000) #extracting most common words


def make_dataset(dictionary):
    direc = "emails/"
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    feature_set = []
    labels = []
    c = len(emails)

    for email in emails:
        data = []
        f = open(email,encoding="iso8859_1")
        words = f.read().split(' ')
        for entry in dictionary:
            #print(entry[0])
            data.append(words.count(entry[0]))
        feature_set.append(data)

        if "ham" in email:
            labels.append(0)
        if "spam" in email:
            labels.append(1)
       # print (c)
        c = c - 1
    return feature_set, labels


d = make_dict()
features, labels = make_dataset(d)
    #NOW TRAINING DATASET 
x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)     #IN X WE HAVE FEATURES IN Y WE HAVE LABELS

clf = LinearSVC()
clf.fit(x_train, y_train)

preds = clf.predict(x_test)
print (accuracy_score(y_test, preds)*100)
save(clf, "text-classifier.mdl")
