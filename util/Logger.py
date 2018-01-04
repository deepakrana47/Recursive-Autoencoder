import pickle

class logger:
    def __init__(self, type, log_filename):
        self.fname = log_filename
        if type == 'pickle':
            self.log_data = self.log_pickle
            self.get_data = self.delog_pickle
        elif type == 'text':
            self.log_data = self.log_text
            open(self.fname, 'w')
        else:
            print "Logger type not correct !!!"
            exit()
        pass

    def log_pickle(self, data):
        pickle.dump(data, open(self.fname,'wb'))

    def delog_pickle(self):
        return pickle.load(open(self.fname,'rb'))

    def log_text(self, text):
        open(self.fname,'a').write(text)
