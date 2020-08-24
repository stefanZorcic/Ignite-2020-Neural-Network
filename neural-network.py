import csv

u=0

temp = [0] * 0
#This array is the machine's answer for the training data
X = [0] * 0
#This array is the right answer for the training data
Y = [0] * 0

inter = 60 

train = 50  # Number of how much of data is test data
#this is how we read in the training data
with open('training_data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        temp.append(row)
#these are all the libraries we are bringing in for this program
import string
import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.corpus import product_reviews_1 as pro
import numpy as np
#downloading specific parts of the library
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('product_reviews_1')

#binary search to find our element within a sorted list of words
def find(L, target):
    #lower bound which is the start of the list
    start = 0
    #upper bound which is the end of the list
    end = len(L) - 1
    #the while statement that will update the start/end for each iteration
    while start <= end:
        middle = (start + end) / 2
        midpoint = L[int(middle)]
        if midpoint > target:
            end = middle - 1
        elif midpoint < target:
            start = middle + 1
        else:
            return midpoint

#stop words, which are words that have no meaning in sentiment analysis, e.g. I, Me, We, Have
stop_words = set(stopwords.words('english'))
#reading in the sentence and tokenizing it
for i in range(inter):
    indices = [0] * 0
    # Text preprocessing
    text = str(temp[i + 1][2])

    if str(temp[i + 1][3]) == "1":
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

    # text = nltk.word_tokenize(text)


    word_list = pro.words()
    word_list = sorted(word_list)
    indices = ['0.01'] * 73833

    # print(text)


    z = 0

    for i in range(int(len(text))):
        try:
            #binary searching to find our item in the lits
            o = (find(word_list, text[i]))
            q = int(indices[o])
            q += 1.0 * i;
            #inserting it into indices
            indices.pop(int(0 - 1))
            indices.insert(int(o - 1), str(q))
            q = 0

        except TypeError:
            #when the item isn't found, we do this
            indices.pop(73833 - 1)
            z += 1.0 * i
            indices.append(str(z))

    #print(len(indices))
    # indices = np.array(indices)
    # print(indices)
    # print(indices)

    u+=1
    print(u)
    #changing the indices list to an array
    indices = np.array(indices)
    indices = indices.astype(np.float64)
    
    X.append(indices)

Y = np.array(Y)

print(Y)

print(Y)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#the mlp learning machine
classifier = MLPClassifier(hidden_layer_sizes=(10, 10),
                           activation='relu',
                           solver='adam',
                           alpha=0.05,
                           batch_size='auto',
                           learning_rate='adaptive',
                           learning_rate_init=0.1,
                           power_t=0.5,
                           max_iter=50,
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
                           max_fun=1500)

counter = 0

'''
s=True
while s:
    for i in range(100):
        X_train=X[i]
        X_train=X_train.astype(np.float64)
        #print(X_train)
        print(counter,classifier.fit([X_train], [Y[i]]))
        counter+=1;
    if(input()=="x"):
        s=False
'''
#training the data
X_train = X
# X_train=X_train.astype(np.float64)
# print(X_train)

print(X_train)
print(Y)

Y = np.array(Y)
Y.shape
print(Y)

classifier.fit(X_train, Y)
counter += 1;
print("YYYAYYAYAYYAYAYYA")

num = 0
dem = 0

for i in range(inter - train):
    X_Test = X[i + train]
    X_Test = X_Test.astype(np.float64)
    # print(str(classifier.predict([X_Test])) + str(Y[i+int(train)]))
    if (int(classifier.predict([X_Test]))) == int(Y[i + int(train)]):
        num += 1
    dem += 1

    print((int(classifier.predict([X_Test]))), int(Y[i + int(train)]))

print(str(num / dem * 100) + "% Accuracy")

'''
p=100;
data=[0]*0
with open('training_data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        data.append(row)
for i in range(inter):
    indices=[0]*0
    # Text preprocessing
    text = str(data[i+1][2])
    text = text.lower()
    remove_digits = str.maketrans('', '', string.digits)
    text = text.translate(remove_digits)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = nltk.word_tokenize(text)
    word_list = words.words()
    indices = ['0.01'] * 236737
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
    data.append(indices)
for i in range(p):
    (int(classifier.predict([X_Test])))

'''
