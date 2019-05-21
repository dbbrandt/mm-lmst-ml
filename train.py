from utils import get_dimensions
from utils import ACCEPTED_CHARS
import numpy as np
import pandas as pd

predictor_col = 'correct_answer'
filename = 'data/names_medium.csv'
category_filename = 'data/unique_answers.csv'

# Define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(ACCEPTED_CHARS))
int_to_char = dict((i, c) for i, c in enumerate(ACCEPTED_CHARS))


# Removes all non accepted characters
def normalize(line):
    return [c.lower() for c in line if c.lower() in ACCEPTED_CHARS]


# Returns a list of n lists with n = word_vec_length
def name_encoding(name):
    char_vec_length, word_vec_length, output_labels = get_dimensions(filename, category_filename, predictor_col)
    # Encode input data to int, e.g. a->1, z->26
    integer_encoded = [char_to_int[char] for i, char in enumerate(name) if i < word_vec_length]

    # Start one-hot-encoding
    onehot_encoded = list()

    for value in integer_encoded:
        # create a list of n zeros, where n is equal to the number of accepted characters
        letter = [0 for _ in range(char_vec_length)]
        letter[value] = 1
        onehot_encoded.append(letter)

    # Fill up list to the max length. Lists need do have equal length to be able to convert it into an array
    for _ in range(word_vec_length - len(name)):
        onehot_encoded.append([0 for _ in range(char_vec_length)])

    return onehot_encoded

def get_target_lookup(category_file):
    categories = pd.read_csv(category_file)
    categories['id'] = categories.index
    categories.set_index('correct_answer', inplace=True)
    return categories

def convert_target(target, category_file):
    print("Converting Target Data")
    categories = get_target_lookup(category_file)

    converted = []
    for i, x in enumerate(target):
        converted.append(categories.loc[x].id)
        if i % 10000 == 0:
            print(i)
    return converted

# Encode the output labels
def lable_encoding(target):
    labels = np.empty((322, 2))
    for i in target:
        if i == 'm':
            labels = np.append(labels, [[1, 0]], axis=0)
        else:
            labels = np.append(labels, [[0, 1]], axis=0)
    return labels

df = pd.read_csv(filename)

# Split dataset in 60% train, 20% test and 20% validation
train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

# Convert both the input names as well as the output lables into the discussed machine readable vector format
train_x = np.asarray([np.asarray(name_encoding(normalize(name))) for name in train[predictor_col]])
train_y = lable_encoding(train.gender)

validate_x = np.asarray([name_encoding(normalize(name)) for name in validate[predictor_col]])
validate_y = lable_encoding(validate.gender)

test_x = np.asarray([name_encoding(normalize(name)) for name in test[predictor_col]])
test_y = lable_encoding(test.gender)