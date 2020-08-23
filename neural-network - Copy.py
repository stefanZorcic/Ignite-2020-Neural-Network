import csv
temp=[0]*0
X = [0]*0
Y = [0]*0

u=0

inter = 500

train = 450 # Number of how much of data is test data


with open('training_data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        temp.append(row)

import string
import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.corpus import product_reviews_1 as pro
import numpy as np

nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('product_reviews_1')

stop_words=set(stopwords.words('english'))

word_list = pro.words()

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

    word_tokens = nltk.word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    text = filtered_sentence

    indices = ['0.01'] * 73833


    z=0

    for i in range(int(len(text))):
        try:
            o = (word_list.index(text[i]))
            q = int(indices[o])
            q+=1.0*i;
            indices.pop(int(0-1))
            indices.insert(int(o-1),str(q))
            q=0

        except ValueError:
            indices.pop(73833-1)
            z+=1.0*i
            indices.append(str(z))

    #print(len(indices))
    #indices = np.array(indices)
    #print(indices)
    #print(indices)

    u+=1
    print(u)

    indices = np.array(indices)
    indices = indices.astype(np.float64)

    X.append(indices)

Y = np.array(Y)

print(Y)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

classifier = MLPClassifier(hidden_layer_sizes=(100,2),
                           activation='relu',
                           solver='lbfgs',
                           alpha=0.05,
                           batch_size='auto',
                           learning_rate='adaptive',
                           learning_rate_init=0.05,
                           power_t=0.5,
                           max_iter=5000,
                           shuffle=True,
                           random_state=None,
                           tol=0.01,
                           verbose=False,
                           warm_start=False,
                           momentum=0.3,
                           nesterovs_momentum=True,
                           early_stopping=False,
                           validation_fraction=0.1,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-08,
                           n_iter_no_change=10,
                           max_fun=15000)

counter=0


X_train=X
print(X_train)
print(Y)

Y=np.array(Y)
Y.shape
print(Y)

classifier.fit(X_train, Y)
counter+=1;
print("YYYAYYAYAYYAYAYYA")




num = 0
dem = 0

for i in range(inter-train):
    X_Test=X[i+train]
    X_Test=X_Test.astype(np.float64)
    #print(str(classifier.predict([X_Test])) + str(Y[i+int(train)]))
    if (int(classifier.predict([X_Test]))) == int(Y[i+int(train)]):
        num+=1
    dem+=1

    print((int(classifier.predict([X_Test]))), int(Y[i+int(train)]))

print(str(num/dem*100)+"% Accuracy")

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

temp=[0]*0
X = [0]*0

u=0


with open('contestant_judgment.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        temp.append(row)

nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('product_reviews_1')

stop_words=set(stopwords.words('english'))

for i in range(600000):
    indices=[0]*0
    # Text preprocessing
    text = str(temp[i+1][2])

    text = text.lower()
    remove_digits = str.maketrans('', '', string.digits)
    text = text.translate(remove_digits)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()

    word_tokens = nltk.word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    text = filtered_sentence

    word_list = pro.words()
    indices = ['0.01'] * 73833

    z=0

    for i in range(int(len(text))):
        try:
            o = (word_list.index(text[i]))
            q = int(indices[o])
            q+=1.0*i;
            indices.pop(int(0-1))
            indices.insert(int(o-1),str(q))
            q=0

        except ValueError:
            indices.pop(73833-1)
            z+=1.0*i
            indices.append(str(z))

    u+=1
    print("Loading: " + str(u))

    indices = np.array(indices)
    indices = indices.astype(np.float64)

    X.append(indices)

print(X_train)


for i in range(inter-train):
    X_Test=X[i+1]
    X_Test=X_Test.astype(np.float64)

    ans = int(classifier.predict([X_Test]))

    print((ans), int(Y[i+i]))

    v = open('contestant_judgment.csv')
    r = csv.reader(v)
    row0 = r.next()
    row0.append('berry')
    for item in r:
        item.append(int(ans))



