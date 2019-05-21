import pandas as pd

ACCEPTED_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'

def get_dimensions(filename, category_filename, predictor_col):
    df = pd.read_csv(filename)
    word_vec_length = min(df[predictor_col].apply(len).max(), 25) # Length of the input vector
    char_vec_length = len(ACCEPTED_CHARS) # Length of the character vector

    dfc = pd.read_csv(category_filename)
    output_labels = len(dfc) # Number of output labels

    print(f"The input vector will have the shape {word_vec_length}x{char_vec_length}.")
    print(f"Output categories: {output_labels}")

    return word_vec_length, char_vec_length, output_labels

def get_target_lookup(category_filename):
    categories = pd.read_csv(category_filename)
    categories['id'] = categories.index
    categories.set_index('correct_answer', inplace=True)
    return categories