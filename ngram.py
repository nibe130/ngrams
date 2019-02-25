import numpy as np
import string
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


#Used to classiy the data
#uses SVM
def classification(cmatrix):
    from sklearn import preprocessing
    normalized_X = preprocessing.normalize(cmatrix)

    from sklearn.model_selection import train_test_split
    from sklearn import svm
    # X = normalized_X
    # y = lables
    # X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.1, random_state=42)
    # clf = svm.SVC(kernel='linear', C=1)
    # from sklearn.linear_model import LogisticRegression
    # # clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto').fit(X_trn, y_trn)
    # from sklearn.model_selection import cross_val_score


    acc=[]
    falsepos=[]
    falseneg=[]
    trueneg=[]
    recall=[]
    precision=[]
    truep=[]
    label_s=lables

    #Kcross vlaidaion code
    for i in range(10):
        ktrain = normalized_X[(i * 100):(100 * (i + 1)), :]
        label_s = np.array(label_s)
        label_s = np.reshape(label_s, (1000, 1))
        tktrain = label_s[(i * 100):(100 * (i + 1)), :]
        tkfold1 = np.zeros((900, normalized_X.shape[1]))
        lblkfold = np.zeros((900, 1))

        for j in range(10):
            if (j != i):
                temp = normalized_X[(j * 100):(100 * (j + 1)), :]

                templbl = label_s[(j * 100):(100 * (j + 1)), :]

                tkfold1 = np.append(tkfold1, temp, 0)

                lblkfold = np.append(lblkfold, templbl, 0)
        tkfold1 = tkfold1[900:,:]
        lblkfold = lblkfold[900:,:]
        # print("fold:",i,tkfold1.shape,lblkfold.shape,ktrain.shape,tktrain.shape)
        clf = svm.SVC(kernel='linear')
        import pandas as pd
        lblkfold = pd.DataFrame(lblkfold)

        clf.fit(tkfold1, lblkfold)
        y_pred = clf.predict(ktrain)
        tn, fp, fn, tp = confusion_matrix(y_pred, tktrain).ravel()
        #print(tn, fp, fn, tp)
        falsepos.append((fp / (fp + tn)))
        falseneg.append((fn / (fn + tp)))
        trueneg.append((tn / (tn + fp)))
        truep.append((tp / (tp + fn)))
        recall.append((tp / (tp + fn)))
        precision.append((tp / (tp + fp)))
        from sklearn.metrics import accuracy_score
        acc.append(accuracy_score(tktrain, y_pred))
    print("false possitive= {}".format(sum(falsepos)/len(falsepos)))
    print("false negative= {}".format(sum(falseneg) / len(falseneg)))
    print("true negative= {}".format(sum(trueneg) / len(trueneg)))
    print("recall= {}".format(sum(recall) / len(recall)))
    print("TP= {}".format(sum(truep) / len(recall)))
    print("precision= {}".format(sum(precision) / len(precision)))
    print("accuracy= {}".format(sum(acc) / len(acc)))








    # scores = cross_val_score(clf, X, y, cv=10)
    # #clf.fit(X_trn, y_trn)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # from sklearn.metrics import accuracy_score
    # # y_pred = clf.predict(X_tst)
    # # print(accuracy_score(y_tst, y_pred))

#remove caps
#remove punctuations
def preprocessing(fname):
    values = []
    sentences=[]
    labels=[]
    raw =open(fname).read().lower().splitlines()

    for i in raw:
        values.append(i.split("\t"))

    for sentence, label in values:
        sentences.append(sentence)
        labels.append(label)

    tokenizer = RegexpTokenizer(r'\w+')
#tokenized and preprocessed sentences
    s_t=[]
    for sentence in sentences:
        s_t.append(tokenizer.tokenize(sentence))

    return s_t,labels

def onegrams(s,lables):
    ul=[]
    for element in s:
        for item in element:
            if item in ul:
                continue
            else:

                ul.append(item)
    ul.sort()
    ocm = np.zeros(shape=(len(s), len(ul)))
    for element in s:
        for item in element:
            if item in ul:
                ocm[s.index(element)][ul.index(item)]+=1
    print("onegrams")
    classification(ocm)
    return ocm


def bigrams(s,lables):
    resultSet=[]
    uniques=[]
    for sentence in s:
        zips=zip(sentence,sentence[1:])
        resultSet.append(list(set(zips)))

    fl1 = [item for sublist in resultSet for item in sublist]

    x=list(set(fl1))

    bcm = np.zeros(shape=(len(resultSet), len(x)))
    len(fl1)
    for item in resultSet:
        for element in item:
            if element in x:
                bcm[resultSet.index(item)][x.index(element)]+=1
    print(len(x))
    print("bigrams")
    classification(bcm)
    return bcm


def trigrams(s,lables):
    resultSet=[]
    uniques=[]
    for sentence in s:
        zips=zip(sentence,sentence[1:],sentence[2:])
        resultSet.append(list(set(zips)))

    fl2 = [item for sublist in resultSet for item in sublist]

    y = list(set(fl2))


    tcm = np.zeros(shape=(len(resultSet), len(y)))

    for item in resultSet:
        for element in item:
            if element in y :
                tcm[resultSet.index(item)][y.index(element)]+=1
    print("trigrams")
    classification(tcm)
    return  tcm

def fgrams(s,lables):
    resultSet = []
    uniques = []
    for sentence in s:
        zips = zip(sentence, sentence[1:], sentence[2:],sentence[3:])
        resultSet.append(list(set(zips)))

    fl3 = [item for sublist in resultSet for item in sublist]

    z = list(set(fl3))

    fcm = np.zeros(shape=(len(resultSet), len(z)))

    for item in resultSet:
        for element in item:
            if element in z:
                fcm[resultSet.index(item)][z.index(element)] += 1
    print("fgrams")
    classification(fcm)
    return fcm

def fivegrams(s,lables):
    resultSet = []
    uniques = []
    for sentence in s:
        zips = zip(sentence, sentence[1:], sentence[2:],sentence[3:],sentence[4:])
        resultSet.append(list(set(zips)))

    fl4 = [item for sublist in resultSet for item in sublist]

    a = list(set(fl4))

    ficm = np.zeros(shape=(len(resultSet), len(a)))

    for item in resultSet:
        for element in item:
            if element in a:
                ficm[resultSet.index(item)][a.index(element)] += 1
    print("fivegrams")
    classification(ficm)
    return ficm


#concatenates the matrixes and does classification
def concat(f1,f2,f3,f4,f5):
    f12=np.concatenate((f1, f2), axis=1)
    f123 = np.concatenate((f1, f2, f3), axis=1)
    f1234=np.concatenate((f1, f2,f3,f4), axis=1)
    f12345=np.concatenate((f1,f2,f3,f4,f5), axis=1)
    print(f12.shape)
    print(f123.shape)
    print(f1234.shape)
    print(f12345.shape)
    classification(f12)
    classification(f123)
    classification(f1234)
    classification(f12345)


file_names=["amazon_cells_labelled.txt","imdb_labelled.txt","yelp_labelled.txt"]
for fname in file_names:
    sentences,lables=preprocessing(fname)
    f1=onegrams(sentences,lables)
    f2=bigrams(sentences,lables)
    f3=trigrams(sentences,lables)
    f4=fgrams(sentences,lables)
    f5=fivegrams(sentences,lables)
    concat(f1,f2,f3,f4,f5)
