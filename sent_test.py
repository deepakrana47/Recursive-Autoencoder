from util.Utility import generate_fixed_vector2
from models.recursive_autoencoder import stack_RAE
from util.Vectorization import Word_vector
from util.Preprocessing import preprocess
import pickle

# model File Name
var_file = 'weights/model_variables.pickle'

# initalizing model
isize, hsize, w, b, g = pickle.load(open(var_file, 'rb'))
nn = stack_RAE(input_size=isize, hidden_size=hsize)
nn.w = w
nn.b = b

# variables
isize = 50
modelFile = '/media/zero/41FF48D81730BD9B/DT_RAE/data/word_embeddings/50/wiki_word50.vector.pickle'
parse_type = 'dep'
strType = 'h'
stp = 0
nfeat = 1
pool_size = 10
clf = pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/Git_DT_RAE/weights/classifier_11.pkl','rb'))

# Initalizing word vector and preprocessor.
wvect = Word_vector(isize)
data_processing = preprocess(parsing_type=parse_type, structure_type=strType, stopword=stp, wvect=wvect)

sent1 = raw_input("Enter Sentence1 : ")
sent2 = raw_input("Enter Sentence2 : ")
# sent1 = "Consumers would still have to get a descrambling security card from their cable operator to plug into the set."
# sent2 = "To watch pay television, consumers would insert into the set a security card provided by their cable service."


wd = data_processing.sentProcess(sent1) + data_processing.sentProcess(sent2)

vects, smin = generate_fixed_vector2(nn, wd, nfeat, pool_size)

out = clf.predict(vects)

print sent1
print sent2+'\n'
print "Result: ", 'Not Paraphrase' if out[0] == 0 else "Paraphrase"
print "\nPhrase Similarity value(0=similar, above 0=dissimilarity index)"
for i in sorted(smin):
    print '\t',smin[i][0],':\t\t',smin[i][1]

