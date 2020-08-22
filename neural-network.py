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
    indices = [0] * 236736
    i=0

    for i in range(int(len(text))):
      try:
        indices[word_list.index(text[i])] += 1
      except ValueError:
        i+=1;
        indices.pop(i)
        indices.append(i)


    X.append(indices)

print(X)
print(Y)


from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(hidden_layer_sizes=(1,1), max_iter=236736,activation = 'relu',solver='adam',random_state=1)

classifier.fit(X, Y)

print(X)
print(len(X[0]))

for i in range(len(X)):
    classifier.predict(X[i])

print("END")


