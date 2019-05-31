import numpy as np
import pandas as pd
from vectorizer import Vectorizer
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import joblib

answer_col = 'answer'
predictor_col = 'correct_answer'
filename = 'data/names_medium.csv'
category_filename = 'data/unique_answers.csv'

def build_and_train(vectorizer, batch_size, epocs, learning_rate, hidden=None):
    print('Build model...')
    label_count = len(vectorizer.categories)
    if hidden:
        hidden_nodes = hidden
    else:
        hidden_nodes = int(2 / 3 * ((vectorizer.word_vec_length * vectorizer.char_vec_length) + label_count))
    print(f"Hidden nodes calulcate: {hidden_nodes}")

    model = Sequential()
    model.add(LSTM(hidden_nodes, return_sequences=False, input_shape=(vectorizer.word_vec_length,
                                                                      vectorizer.char_vec_length)))
    model.add(Dropout(0.2))
    model.add(Dense(units=label_count))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate, decay=learning_rate / epocs),
                  # optimizer='adam',
                  metrics=['acc'])

    train_x, train_y, validate_x, validate_y, test_x, test_y = vectorizer.get_datasets()
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epocs, validation_data=(validate_x, validate_y),
              callbacks=[early_stopping])
    return model

def check_prediction(pred_y, target):
    pred_y = pred_y.round()
    y_int = np.where(pred_y == 1)
    predicted_category = "None"
    if len(y_int[1]) > 0:
        y_int = y_int[1][0]
        predicted_category = vectorizer.int_to_target[y_int]
    correct = predicted_category == target
    return correct, predicted_category

def interactive_test(filename, vectorizer):
    categories = vectorizer.categories
    model_filename = filename + '.joblib'
    model = joblib.load(model_filename)
    for i in range(0,100):
        category = categories[int(np.random.uniform(0, len(categories), 1))]
        print('Enter misspelling for: {}'.format(category))
        string = input('Input data: ')
        data_x, data_y = vectorizer.format_single(string, category)
        pred_y = model.predict(data_x)
        correct, predicted_category = check_prediction(pred_y, category)
        if correct:
            print(f"Correct!!!!")
        else:
            print(f"Incorrect.... Predicted {predicted_category}.")

def test_list(model, vectorizer, predictors, targets):
    categories = vectorizer.categories
    correct_count = 0
    for i, x in enumerate(predictors):
        target = targets[i]
        data_x, data_y = vectorizer.format_single(x, target)
        pred_y = model.predict(data_x)
        correct, precdicted_category = check_prediction(pred_y, target)
        if correct:
            correct_count += 1
        else:
            print(f"Error: Input: {x} Expected: {target} got {precdicted_category}.")
    correct_pct = correct_count / len(predictors) * 100
    print(f"Result of category test: {correct_pct}%")

def test_manual_data(filename, test_cases, vectorizer):
    model_filename = filename + '.joblib'
    model = joblib.load(model_filename)

    validate(model, vectorizer)

    targets = vectorizer.categories
    test_list(model, vectorizer, targets, targets)
    predictors = np.array(test_cases)[:, 0:1].flatten()
    targets = np.array(test_cases)[:, 1:2].flatten()
    test_list(model, vectorizer, predictors, targets)

def validate(model, vectorizer):
    pred_y = model.predict(vectorizer.test_x)
    pred_y = pred_y.round()
    score = accuracy_score(vectorizer.test_y, pred_y)
    print(f"Accuracy score = {score}")
    return score

def main(filename, category_filename, answer_col, predictor_col, hidden_nodes):
    df = pd.read_csv(filename, usecols=[answer_col, predictor_col])
    categories = pd.read_csv(category_filename, usecols=[predictor_col])[predictor_col].values
    vectorizer = Vectorizer(df, categories, predictor_col, answer_col)
    vectorizer.format(0.6, 0.2)

    batch_size = 1000
    epocs = 50
    learning_rate = 1e-3
    model = build_and_train(vectorizer, batch_size, epocs, learning_rate, hidden_nodes)
    validate(model, vectorizer)
    joblib.dump(model, filename + '.joblib')

# main(filename, category_filename, answer_col, predictor_col, 600)

df = pd.read_csv(filename, usecols=[answer_col, predictor_col])
categories = pd.read_csv(category_filename, usecols=[predictor_col])[predictor_col].values
vectorizer = Vectorizer(df, categories, predictor_col, answer_col)
vectorizer.format(0.6, 0.2)
# interactive_test(filename, vectorizer )

test_cases = [['Amie Adams', 'Amy Adams'],
              ['Michael Fox', 'Michael J. Fox'],
              ['Minny Driver', 'Minnie Driver'],
              ['BLair Underwood', 'Blair Underwood'],
              ['Ralph Finnes', 'Ralph Fiennes'],
              ['Kate Blanchette', 'Cate Blanchett'],
              ['Joakin Pheonix', 'Joaquin Phoenix'],
              ['Ane Hathaway', 'Anne Hathaway'],
              ['Mickey Rorke', 'Mickey Rourke'],
              ['Collin Farrell', 'Colin Farrell'],
              ['Ben Stiler', 'Ben Stiller'],
              ['Cate Winslet', 'Kate Winslet'],
              ['John Hawks', 'John Hawkes'],
              ['George Cloney', 'George Clooney'],
              ['Cathlene Turner','Kathleen Turner'],
              ['Mathew Broderick', 'Matthew Broderick'],
              ['Mat Damon', 'Matt Damon'],
              ['Jennifer Jason Lee', 'Jennifer Jason Leigh'],
              ['Peter Otolle', "Peter O'toole"],
              ['John C Reily', 'John C. Reilly'],
              ["Elisabeth Perkins","Elizabeth Perkins"],
              ["Mark Walburg", "Mark Wahlberg"]
             ]

test_manual_data(filename, test_cases, vectorizer)