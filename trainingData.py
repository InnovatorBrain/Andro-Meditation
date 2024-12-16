import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Download necessary NLTK resources if not already present
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

# Tokenize the patterns and create a list of words and classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

# Create the bag of words and output row
# Create the bag of words and output row
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Create the bag of words for the current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    # Ensure that bag and output_row have the same dimensions
    assert len(bag) == len(words), f"Mismatch in bag length: {len(bag)} != {len(words)}"
    assert len(output_row) == len(classes), f"Mismatch in output row length: {len(output_row)} != {len(classes)}"

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)  # Use dtype=object to handle lists of different lengths

# Split training data into input (X) and output (Y)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbotmodel.h5')

print('Done')
