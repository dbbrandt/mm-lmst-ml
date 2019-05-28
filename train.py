import pandas as pd
from vectorizer import Vectorizer


answer_col = 'answer'
predictor_col = 'correct_answer'
filename = 'data/names_medium.csv'
category_filename = 'data/unique_answers.csv'

def main(filename, category_filename, answer_col, predictor_col):
    df = pd.read_csv(filename, usecols=[answer_col, predictor_col])
    categories = pd.read_csv(category_filename, usecols=[predictor_col])[predictor_col].values
    vectorizer = Vectorizer(df, categories, predictor_col, answer_col)
    train_X, train_y, validate_X, validate_Y, test_X, test_y = vectorizer.format(0.6, 0.2)


main(filename, category_filename, answer_col, predictor_col)

