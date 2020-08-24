
#These are the modulues that will be used in the program
import csv #CVS is imported to manipulate the data with the cvs files
import string #String is imported to clean up the data input from the cvs files
import nltk # The Natural Language tool kit used to include the English Dictionary of words
from nltk.corpus import words #words is imported as a library of all the english words in the english alphabet for future reference
from nltk.corpus import stopwords #stopswords is imported as a library that has a prebuilt dict of stop words to clean up cvs file input
from nltk.corpus import product_reviews_1 as pro #Product review is imported as a dictionary of words that are common in positive and negative description, imported as pro for ease
import numpy as np # numpy is imported to deal with complex caluclationg, imported as np due to common notation
from sklearn.neural_network import MLPClassifier #MLPClassifier is imported as the brains of the neural network, MLP is used over over method due to the high customability even though there is a trade of for speed
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score #Common modules from sklearn that are used as metrics are imported
from csv import writer #writer is imported for latter use to write contestant_judgment file
from csv import reader #reader is used as an optimized way to read contestant_judgment file


temp=[0]*0 #Temp is a matrix holds the training data in line 34, to be used latter

# Positive or negative verdict
X = [0]*0 # Is defined as a matrix in Line 134
Y = [0]*0

u=0 #u is a counter to see how many lines of the training data has been processed into the input layer of the neural network

inter = 1000000 # Number of lines of training_data.cvs file is being used, 1 million is the maximum

train = 950000 # Number of lines will be deticated towards training, the remaing lines will be used to measure accuracy of neural network, train < inter

with open('training_data.csv') as csvDataFile: # Read in the training data through the CSV file
    csvReader = csv.reader(csvDataFile) #uses the cvs module and the reader subbranch to read the file 

    # Read in each row of the input
    for row in csvReader:
        temp.append(row) #Temp (line 16) holds the data of the cvs file in a matrix

# Download all nltk required corpus for this program
nltk.download('words') #All the words in the enghlish dictionary
nltk.download('punkt') #Words to sanitize training_data input
nltk.download('stopwords') #prebuilt dictionary of nltk stop words
nltk.download('product_reviews_1') #prebuilt dictionary of word commonly used in product review this is important in establishing important words in the training data input


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

#word list is defined as the dicitionary of pro (Line 8)
word_list = pro.words()

#reading in the sentence and tokenizing it
for i in range(inter):
    indices=[0]*0 #indices is reset after each loop, so previous data does not affect future data

    # Text preprocessing
    text = str(temp[i+1][2])

    #Determines if a given string from training_data.cvs is positive or negative
    if str(temp[i+1][3])=="1":
        Y.append(1) #Y (Line 20) determines the correct answer of a given string in the training_data.cvs data 
    else:
        Y.append(0) #Integer 0 states a negative phrase will 1 means a positive phrase

    # Clean up the text (removing special characters and turning characters into lower case)
    text = text.lower() # make the text all lower case, to make the input consiestent
    remove_digits = str.maketrans('', '', string.digits) # remove_digits shows the digits to remove
    text = text.translate(remove_digits) #removes all the digits from the text
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove all punctuation
    text = text.strip() # Trim the string of leading and trailing spaces

    # Tokenize the string into separate words
    word_tokens = nltk.word_tokenize(text)

    # Filter the sentence of all stop words
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = [] #filtered sentence is sanitized for use in line 102

    for w in word_tokens: #Loops for the tokens in the text string
        if w not in stop_words: #Checks string for stop words
            filtered_sentence.append(w) #iltered_sentence is the final cleaned text output

    text = filtered_sentence #sets array text to be filitered_sentence for easier future notation

    # The index list for all input words
    indices = ['0.01'] * 73833

    
    z=0 #z is a counter for unkown dictonary words, is reset every loop to be cleaned for next loop

    for i in range(int(len(text))): #loops for the number of words in the toknzied text 
        try: #tries to do a binary search of item is word_list
            #binary searching to find our item in the lits
            o = (find(word_list, text[i]))
            q = int(indices[o])
            q += 1.0 * i; 
            #inserting it into indices
            indices.pop(int(0 - 1))
            indices.insert(int(o - 1), str(q)) #converts the word into base 10 integer as a function of frequency and position
            q = 0

        except TypeError: #Type Error if the word is not found in the dictonary
            indices.pop(73833 - 1)
            z += 1.0 * i
            indices.append(str(z)) #converts the word into base 10 integer as a function of frequency and position

    u+=1 #u (Line 22) counter is increased by 1
    print(u) #print which line of the data that is being converted into and index for the input layer of the neural network

    indices = np.array(indices) #indices is converted into an array from a list
    indices = indices.astype(np.float64) #the indice is converted from integers to float 64(bit) points

    X.append(indices) # X (Line 19) holds the matrix of indices for latter use in the neural network as training data

Y = np.array(Y) # Y is shaped as an array from the numpy module

print(Y) #Prints Y to show the data set, this is used to show that the code is working up to this point and to make sure the data has diversity

#MLPClassifier is the main logic of the neural network and is defined by the paramenters (Line 141-163)
classifier = MLPClassifier(hidden_layer_sizes=(10,2), #2 hidden layers is defined
                           activation='relu', #Relu is chosen for the low computing cost
                           solver='lbfgs', # LBFGS is chosen as the solver since the train data is relitively small (1 million pieces)
                           alpha=0.05, #Alpha is set to 0.05, this values has shown the best results in testing
                           batch_size='auto', #batch sizing is set to auto
                           learning_rate='adaptive', #Learning rate is set to adaptive, meaning as long as the cost function is decreasing the learning rate will not change
                           learning_rate_init=0.05, # common value
                           power_t=0.5, # common value
                           max_iter=10000000, #Chosen not to have a buffer overflow
                           shuffle=True, # common value
                           random_state=None, # common value
                           tol=0.01, #determined through testing
                           verbose=False, # common value
                           warm_start=False, # common value
                           momentum=0.3,#determined through testing
                           nesterovs_momentum=True, # common value
                           early_stopping=False, # common value
                           validation_fraction=0.1, #determined through testing
                           beta_1=0.9, #determined through testing
                           beta_2=0.999, #determined through testing
                           epsilon=1e-08, #determined through testing
                           n_iter_no_change=10, #determined through testing
                           max_fun=1500000) #determined through testing

counter=0 #counter is set to 0 for future use

# Training the data manipulation
X_train=X
print(X_train)
print(Y)

#The answer to the training data manipulation
Y=np.array(Y)
Y.shape
print(Y)

classifier.fit(X_train, Y) #trains the nueral network
counter+=1; #counter is increased by 1 for future use

# An accuracy counter which will determine accuracy of the machine later on
num = 0 #numerator stands for num, used to calculate practical accuracy
dem = 0 #denominator stands for dem, used to calculate practical accuracy

# Determine verdics for each of the input data
for i in range(inter-train):
    X_Test=X[i+train]
    X_Test=X_Test.astype(np.float64)
    print(str(classifier.predict([X_Test])) + str(Y[i+int(train)]))
    if (int(classifier.predict([X_Test]))) == int(Y[i+int(train)]):
        num+=1
    dem+=1

    print((int(classifier.predict([X_Test]))), int(Y[i+int(train)])) #prints 2 numbers for viewing purposes, the first number is the machine guess and the second the actual answer, this is good for see the neural network faults and adjust parameters

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

nltk.download('words') #import nltk modules are updated
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('product_reviews_1')

stop_words=set(stopwords.words('english')) #stop words are defined to the english dictonary

for i in range(10):
    indices=[0]*0
    # Text preprocessing
    text = str(temp[i+1][2])

    text = text.lower() #lowers all the text
    remove_digits = str.maketrans('', '', string.digits)
    text = text.translate(remove_digits) #removes digits
    text = text.translate(str.maketrans('', '', string.punctuation)) #removes punctuation
    text = text.strip() #removes white spaces

    word_tokens = nltk.word_tokenize(text) #uses one tokenzer to tokenize text

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = [] #reset fpr Line 235

    for w in word_tokens: #removes stop words and tokenizes text
        if w not in stop_words:
            filtered_sentence.append(w)

    text = filtered_sentence #text set as filtered_sentence for ease of notation

    word_list = pro.words() #word_list is initilized
    indices = ['0.01'] * 73833 #indices is setup

    z=0 #counter z is reset after everyloop

    for i in range(int(len(text))): #loops for the lenght of the tokenized text in the index
        try: #tries to find the strng the the dictonary
            o = (word_list.index(text[i]))
            q = int(indices[o])
            q+=1.0*i;
            indices.pop(int(0-1))
            indices.insert(int(o-1),str(q)) #inserts the base 10 data to the indices
            q=0 #resets counter q

        except ValueError: #in the exeception of string not found in the dictonary
            indices.pop(73833-1)
            z+=1.0*i
            indices.append(str(z)) #appends the base 10 data to the indices

    u+=1 #counter is increased by one
    print("Loading: " + str(u)) #prints what line of the data is being processed

    indices = np.array(indices) #converts the list into an array
    indices = indices.astype(np.float64) #converts data into 64 bit float

    X.append(indices) #converts the arrays into a matrix for the input layer of the neural network

for i in range(600000): #loops the amount of data within the contestant_judgement.cvs file
    X_Test=X[i] #X_Test is defined as a array within the X matrix
    X_Test=X_Test.astype(np.float64) #converts the data points into 64 bit floats

    ans = int(classifier.predict([X_Test])) #runs the values through the trained neural network

    with open('contestant_judgment1.csv', 'r') as read_obj, \ #opens the contestant_judgement.cvs file IMPORTANT: RENAMED TO contestant_judgement1.cvs
            open('contestant_judgement.cvs', 'w', newline='') as write_obj: #creates the output file
        csv_reader = reader(read_obj) #uses the cvs reader module to parse the data
        csv_writer = writer(write_obj) #uses the cvs writer module to write future data (Line 278)
        for row in csv_reader: #For loop to run through the entire cvs file
            row.append(ans) #appends the machine estiminate to variable row to be entered in Line 278
            csv_writer.writerow(row) #writes the machine estimate
