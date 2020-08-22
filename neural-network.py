import csv
temp=[0]*0
X = [0]*0
Y = [0]*0

with open('demo_data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        temp.append(row)




import string
import nltk
from nltk.corpus import words
import numpy as np

nltk.download('words')
nltk.download('punkt')

for i in range(2):
    indices=[0]*0
    # Text preprocessing
    text = str(temp[i+1][0])

    if str(temp[i+1][1])=="positive":
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
    indices = ['0'] * 236737

    print(text)


    z=0

    for i in range(int(len(text))):
        try:
            o = (word_list.index(text[i]))
            q = int(indices[o])
            q+=1;
            indices.pop(int(0-1))
            indices.insert(int(o-1),str(q))
            q=0

        except ValueError:
            indices.pop(236737-1)
            z+=1
            indices.append(str(z))

    indices = np.array(indices)
    #print(indices)
    #print(indices)

    X.append(indices)


Y = np.array(Y)
Y = np.reshape(Y,(-1,1))


X = np.reshape(X[0],(-1,1))
X1 = np.reshape(X[1],(-1,1))

Y = Y.shape

X = X.shape
X1 = X1.shape

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(hidden_layer_sizes=(1,1), max_iter=236736,activation = 'relu',solver='adam',random_state=1)

print(classifier.fit([X], [Y[0]]))

print(classifier.predict([X1]))
