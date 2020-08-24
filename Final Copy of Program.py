import csv

# The list of input words
temp=[0]*0

# Positive or negative verdict
X = [0]*0
Y = [0]*0

u=0 # (Stefan comment this)

inter = 10 # Number of lines of testing datasets

train = 5 # Number of lines of training data

# Read in the training data through the CSV file
with open('training_data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    # Read in each row of the input
    for row in csvReader:
        temp.append(row)

#All the libraries we are bringing in for this program
import string
import nltk # The Natural Language tool kit used to include the English Dictionary of words
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.corpus import product_reviews_1 as pro
import numpy as np

# Download all modules
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('product_reviews_1')


# In this section we will create a frequency array for all the words in the input. Each word will
# have its own cell in the array
# We will first sort the array then use binary search for each tokenized word to update the word count

# Binary search to find our element within a sorted list of words
def find(L, target):
    #lower bound which is the start of the list
    start = 0
    #upper bound which is the end of the list
    end = len(L) - 1
    #the while statement that will update the start/end for each iteration
    while start <= end:
        middle = (start + end) / 2
        midpoint = L[int(middle)]

        # Test to see where the current guess is compared to the desired target
        if midpoint > target:
            end = middle - 1
        elif midpoint < target:
            start = middle + 1
        else:
            return midpoint

# "Stop words", which are words that have no meaning in sentiment analysis, e.g. I, Me, We, Have
# These words will be set at a significantly lower value
stop_words=set(stopwords.words('english'))

word_list = pro.words()

#reading in the sentence and tokenizing it
for i in range(inter):
    indices=[0]*0

    # Text preprocessing
    text = str(temp[i+1][2])

    if str(temp[i+1][3])=="1":
        Y.append(1)
    else:
        Y.append(0)

    # Clean up the text (removing special characters and turning characters into lower case)
    text = text.lower() # make the text all lower case
    remove_digits = str.maketrans('', '', string.digits) # Remove all numbers
    text = text.translate(remove_digits)
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove all punctuation
    text = text.strip() # Trim the string of leading and trailing spaces

    # Tokenize the string into separate words
    word_tokens = nltk.word_tokenize(text)

    # Filter the sentence of all stop words
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    text = filtered_sentence

    # The index list for all input words
    indices = ['0.01'] * 73833


    z=0

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

# include sklearn neural network modules
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#the mlp learning machine
classifier = MLPClassifier(hidden_layer_sizes=(10,2), # parameters to be entered into the machine
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

# Training the data
X_train=X
print(X_train)
print(Y)

Y=np.array(Y)
Y.shape
print(Y)

classifier.fit(X_train, Y)
counter+=1;

# An accuracy counter which will determine accuracy of the machine later on
num = 0
dem = 0

# Determine verdics for each of the input data
for i in range(inter-train):
    X_Test=X[i+train]
    X_Test=X_Test.astype(np.float64)
    print(str(classifier.predict([X_Test])) + str(Y[i+int(train)]))
    if (int(classifier.predict([X_Test]))) == int(Y[i+int(train)]):
        num+=1
    dem+=1

    print((int(classifier.predict([X_Test]))), int(Y[i+int(train)]))

# Ouput the machine accuracy based on answers
print(str(num/dem*100)+"% Accuracy")

temp=[0]*0
X = [0]*0

u=0

# Write answers to CSV file
with open('contestant_judgment.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        temp.append(row)

nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('product_reviews_1')

stop_words=set(stopwords.words('english'))

for i in range(10):
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

from csv import writer
from csv import reader

for i in range(inter-train):
    X_Test=X[i]
    X_Test=X_Test.astype(np.float64)

    ans = int(classifier.predict([X_Test]))

    with open('contestant_judgment1.csv', 'r') as read_obj, \
            open('output_1.csv', 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        for row in csv_reader:
            row.append(ans)
            csv_writer.writerow(row)
