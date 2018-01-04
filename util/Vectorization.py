import pickle, numpy as np

class Word_vector:

    def __init__(self, vector_size):
        self.nov_count = 0
        self.v_size = vector_size
        if vector_size == 200:
            self.model_file = '../word_vector/vector200.pickle'
            self.extra = '../word_vector/extra200.pickle'
        else:
            self.model_file = '../word_vector/msrp_vector50.pickle'
            self.extra = '../word_vector/extra50.pickle'

        self.word_vec = pickle.load(open(self.model_file, 'rb'))
        self.word_vec1 = pickle.load(open(self.extra, 'rb'))

    def generate_word_vect(self):
        return np.random.normal(0.0, 0.01, (self.v_size, 1))

    def update_word_vect(self):
        pickle.dump(self.word_vec1, open(self.extra, 'wb'))

    def get_word_vect(self, word):
        if word in self.word_vec:
            return self.word_vec[word].reshape((self.v_size, 1))
        elif word in self.word_vec1:
            return self.word_vec1[word].reshape((self.v_size, 1))
        else:
            self.nov_count+=1
            a = self.generate_word_vect()
            self.word_vec1[word] = a
            self.update_word_vect()
            return a
    def set_word_vect(self, word, vect):
        if word in self.word_vec1:
            self.word_vec1[word] = vect
        else:
            self.word_vec[word] = vect

    def save_vector(self,fvect):
        pickle.dump([self.word_vec, self.word_vec1], open(fvect,'wb'))

    def load_vector(self, fvect):
        self.word_vec, self.word_vec1 = pickle.load(open(fvect,'rb'))
