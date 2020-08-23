import csv

# input from the datasets
temp=[0]*0
#machine's guess
X = [0]*0
# answers to the test data
Y = [0]*0


# This is total lines of sentiment data
inter = 10

train = 5    # Number of how much of data is test data

# Reading in the CSV file
with open('training_data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        temp.append(row) # read in a row of input from the CSV file

# Importing the libraries needed for sentiment analysis
import string
# Importing the Natural language toolkit module
import nltk
from nltk.corpus import words
import numpy as np

# download the modules
nltk.download('words')
nltk.download('punkt')
# Processing each sentence in the data, and teaching the machine
for i in range(inter):
    indices=[0]*0
    # Text preprocessing
    text = str(temp[i+1][2])
    #answer to the text
    if str(temp[i+1][3])=="1":
        Y.append(1)
    else:
        Y.append(0)
    # Cleaning the text and tokenizing it
    text = text.lower() # changing all characters to lower case
    remove_digits = str.maketrans('', '', string.digits) # remove all non-alphabetical characters
    text = text.translate(remove_digits)
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuation
    text = text.strip() # trim the string
    text = nltk.word_tokenize(text) # tokenize the input string into words

    # words in the english dictionary
    word_list = words.words()
    # indices the words in the english dictionary
    indices = ['0.01'] * 236737



    z=0
    #checking the number of times a word appears, and then incrementing it in our indices
    for i in range(int(len(text))): # loop through the input list
        #if word is in dictionary, increase its value in the indices
        try:
            o = (word_list.index(text[i]))
            q = int(indices[o])
            #position times frequency
            q+=1*i;
            indices.pop(int(0-1))
            indices.insert(int(o-1),str(q))
            q=0
        #If word isn't in dictionary, add to the back
        except ValueError:
            indices.pop(236737-1)
            #position times frequency
            z+=1*i
            indices.append(str(z))
    #indices list is turned into an array
    indices = np.array(indices)

    X.append(indices)
#chaging the list of answers to an array of anaswer
Y = np.array(Y)

print(Y)

print(Y)
#The machine learning library!
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#defining of the neural network, E.G. Number of hidden layers and neurons per layer
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

# Training the machine to learn to detect sentiment
for i in range(train): # loop through all the training datasets
    X_train=X[i]
    X_train=X_train.astype(np.float64)
    print(X_train)
    print(classifier.fit([X_train], [Y[i]]))

num = 0
dem = 0

# Predict sentiment verdict
for i in range(inter-train):
    X_Test=X[i+train]
    X_Test=X_Test.astype(np.float64)
    #print(str(classifier.predict([X_Test])) + str(Y[i+int(train)]))
    if (int(classifier.predict([X_Test]))) == int(Y[i+int(train)]):
        num+=1
    dem+=1

# Output program accuracy as a percentage
print(str(num/dem*100)+"% Accuracy")
