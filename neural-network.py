import csv
temp=[0]*0
X = [0]*0
Y = [0]*0



inter = 10

train = 5    # Number of how much of data is test data


with open('training_data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        temp.append(row)

import string
import nltk
from nltk.corpus import words
import numpy as np

nltk.download('words')
nltk.download('punkt')

for i in range(inter):
    indices=[0]*0
    # Text preprocessing
    text = str(temp[i+1][2])

    if str(temp[i+1][3])=="1":
        Y.append(1)
    else:
        Y.append(0)

    text = text.lower()
    remove_digits = str.maketrans('', '', string.digits)
    text = text.translate(remove_digits)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = nltk.word_tokenize(text)


    word_list = words.words()
    indices = ['0.01'] * 236737

    #print(text)


    z=0

    for i in range(int(len(text))):
        try:
            o = (word_list.index(text[i]))
            q = int(indices[o])
            q+=1*i;
            indices.pop(int(0-1))
            indices.insert(int(o-1),str(q))
            q=0

        except ValueError:
            indices.pop(236737-1)
            z+=1*i
            indices.append(str(z))

    indices = np.array(indices)
    #print(indices)
    #print(indices)

    X.append(indices)

Y = np.array(Y)

print(Y)

print(Y)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

classifier = MLPClassifier(hidden_layer_sizes=(10,1 ),
                           activation='tanh',
                           solver='adam',
                           alpha=0.0001,
                           batch_size='auto',
                           learning_rate='constant',
                           learning_rate_init=0.001,
                           power_t=0.5,
                           max_iter=236737,
                           shuffle=True,
                           random_state=None,
                           tol=0.0001,
                           verbose=False,
                           warm_start=False,
                           momentum=0.9,
                           nesterovs_momentum=True,
                           early_stopping=False,
                           validation_fraction=0.1,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-08,
                           n_iter_no_change=10,
                           max_fun=15000)


for i in range(train):
    X_train=X[i]
    X_train=X_train.astype(np.float64)
    print(X_train)
    print(classifier.fit([X_train], [Y[i]]))

num = 0
dem = 0

for i in range(inter-train):
    X_Test=X[i+train]
    X_Test=X_Test.astype(np.float64)
    #print(str(classifier.predict([X_Test])) + str(Y[i+int(train)]))
    if (int(classifier.predict([X_Test]))) == int(Y[i+int(train)]):
        num+=1
    dem+=1

print(str(num/dem*100)+"% Accuracy")
