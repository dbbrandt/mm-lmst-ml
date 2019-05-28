# # Ideas and basic algorithm taken from a blog on "Choosing the right Hyperparameters for a simple LSTM using Keras"
# # https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
import numpy as np
from scipy.sparse import dok_matrix

class Vectorizer:

    ACCEPTED_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'
    MAX_LEN = 25

    def __init__(self, df, categories, predictor_col, answer_col, max_len=MAX_LEN):
        self.df = df
        self.categories = categories
        self.predictor_col = predictor_col
        self.answer_col = answer_col
        self.max_len = max_len
        self.chars = Vectorizer.ACCEPTED_CHARS
        self.char_to_int = {}
        self.int_to_char = {}
        self.load_char_conversions()
        self.target_to_int = {}
        self.int_to_target = {}
        self.load_target_conversions()
        self.load_vector_lengths()


    def set_accepted_chars(self, accepted_chars):
        """Set the character set to use for vectorization of the input data
           :param accpted_chars: The string of chars that are to be used
           :return A tuple of the word and charcter vector lengths.
        """
        self.chars = accepted_chars
        self.load_char_conversions()
        return self.word_vec_length, self.char_vec_length

    def load_char_conversions(self):
        """Create the dictionaries used to convert letters to and from an integeger representation (0 based)"""
        self.char_to_int = self.char_to_int or dict((c, i) for i, c in enumerate(self.chars))
        self.int_to_char = self.char_to_int or dict((i, c) for i, c in enumerate(self.chars))

    def load_target_conversions(self):
        self.target_to_int = self.target_to_int or dict((c, i) for i, c in enumerate(self.categories))
        self.int_to_target = self.int_to_target or dict((i, c) for i, c in enumerate(self.categories))


    # This get's the size of an answer character vecor (char_vec_length)
    # and the number of character vectors in the predictor_column based on the largest on found <= max_len
    def load_vector_lengths(self):
        """Calculate the size of an data vector which is a conversion of the character data from the datafrarm
           char_vec_length - the size of array representing a single character
           word_vector_length - the number of charactor vectors for each row <= max_len
        """
        self.word_vec_length = min(self.df[self.predictor_col].apply(len).max(), self.max_len)
        self.char_vec_length = len(self.chars)  # Length of the character vector
        print(f"Format of input vector will be ({self.word_vec_length, self.char_vec_length}.")

    def normalize(self, line):
        """Remove all non accepted characters"""
        return [c.lower() for c in line if c.lower() in self.chars]

    def split_dataset(self, train_pct = 0.6, test_pct = 0.2):
        """Returns three DataFrames split based on the input into train, validation and test

        :param  train_pct: The percentage of the dataset to be used for taining
                test_pct: The percentage to be used for testing (must be less than train_pct
                Implied validation_pct is the remaining data after train and test are satisfied
        :return: train, validate, test
                 Three DataFrames split per the params:
        """
        if not (test_pct < train_pct and test_pct < (1 - train_pct)):
            raise Exception("Invalid paramters: required - train_pct > test and train + test < 1")

        train_split = train_pct
        val_split = 1 - test_pct
        # Split: train = 0 -> train_split, validation = train_pct -> val_split, test = val_split -> 1
        # The method DataFrame.sample randomizes a subset of the data. Using frac=1 effectively randomized the entire set
        # Note: nd.split on DataFrame returns a dataframe
        train, validate, test = np.split(self.df.sample(frac=1), [int(train_split * len(self.df)),
                                                                  int(val_split * len(self.df))])
        return {'train': train, 'validate': validate, 'test': test}

    def answer_encoding(self, string):
        """Returns a list of n lists with n = word_vec_length
           Each character vector will have only one value set to 1 representing the character
           ex. [[0.,0.,1,...0.],
                ...
                [0.,1.,0,...0.]]
           :param The string to be encoded
           :return The encoded string
        """
        # Encode input data to int, e.g. a->1, z->26
        integer_encoded = [self.char_to_int[char] for i, char in enumerate(string) if i < self.word_vec_length]

        # Start one-hot-encoding
        onehot_encoded = list()

        for value in integer_encoded:
            # create a list of n zeros, where n is equal to the number of accepted characters
            letter = [0 for _ in range(self.char_vec_length)]
            letter[value] = 1
            onehot_encoded.append(letter)

        # Fill up list to the max length. Lists need do have equal length to be able to convert it into an array
        for _ in range(self.word_vec_length - len(string)):
            onehot_encoded.append([0 for _ in range(self.char_vec_length)])

        return onehot_encoded

    def lable_encoding(self, target):
        """Encode the output labels as a vector
           The result is a 2-d array, one row for each target (rows of letter encoding)
           :param The target dataset (e.x. train_y, validate_y of test_y)
           :return An array of target data where each vector has a one in the appropriate column for
           the category of that target.
        """
        # Use sparse data format to efficiently load the array with only one significant value per row.
        print(f"Converting targets.")
        sparse_data = dok_matrix((len(target), len(self.categories)))

        for i, value in enumerate(target):
            target_int = self.target_to_int[value]
            sparse_data[i, target_int] = 1
        return sparse_data.toarray()

    def format(self, train_pct, test_pct):
        """Convert both the input names as well as the output lables into vector format
           :param  train_pct - percentage of data to use for the train split
                   test_pct - percentage for the test split
                   implied: verify will be the remainder
           :return The vectorized data in a dictionary of dictionarys.
           The outer dictoionary is the split type (train, validate, test). The inner dictionary is the x and y data.
        """
        split_result = self.split_dataset(train_pct, test_pct)

        results = ()
        for key, data in split_result.items():
            print(f"Formating {key} data. Size = {len(data)}")
            data_x = np.asarray([np.asarray(self.answer_encoding(self.normalize(answer)))
                                 for answer in data[self.answer_col]])
            data_y = self.lable_encoding(data[self.predictor_col])
            results = results + (data_x, data_y)

        return results
