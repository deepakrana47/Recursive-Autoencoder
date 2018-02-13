from models.recursive_autoencoder import stack_RAE
from util.Preprocessing import preprocess
from util.Vectorization import Word_vector
from util.Utility import get_msrp_data, get_results, generate_fixed_vector

import warnings, numpy as np, pickle, os
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from sklearn.neural_network import MLPClassifier
from sklearn import svm

def test_fun(var_file, pool_size, num_feature, stp, parse_type):

    #create result_directory
    clfFile = 'weights/classifier_'+str(num_feature)+str(stp)+'.pkl'
    res_dir = '/'.join(var_file.split('/')[:-1]) + '/results/'
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    if os.path.isfile(clfFile):
        clf = pickle.load(open(clfFile,'rb'))
    else:
        # getting data that is processed for testing porpose
        train, train_label, test, test_label = get_msrp_data(stp)

        #initalizing model
        isize, hsize, w, b, g = pickle.load(open(var_file, 'rb'))
        nn = stack_RAE(input_size=isize, hidden_size=hsize)
        nn.w = w
        nn.b = b
        # a.optimum.opt_var.g=g

        # preprocessing for train and test set
        wvect = Word_vector(isize)
        data_processing = preprocess(parsing_type=parse_type, structure_type='h', stopword=stp, wvect=wvect)
        train_set, _ = data_processing.process_words_data(train)
        test_set, _ = data_processing.process_words_data(test)

        # generating fixed size phrase vector for train and test set
        otrain = generate_fixed_vector(nn, train_set, num_feature, pool_size)
        otest = generate_fixed_vector(nn, test_set, num_feature, pool_size)

        # classifier defination
        # clf = MLPClassifier(activation='logistic', solver='adam',alpha=0.0001,batch_size='auto',learning_rate='adaptive',max_iter=10000,tol=1e-5, verbose=0)
        #
        # clf = svm.LinearSVC(penalty='l1',tol=0.001, C=1.0, loss='hinge', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=7, max_iter=100000)

        clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.00001, C=1.0, multi_class='ovr',
                                    fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0,
                                    random_state=None, max_iter=1000)

        # performing classifier training
        clf.fit(otrain, train_label)
        pickle.dump(clf, open(clfFile,'wb'))
    # performing pridection
    score = clf.predict(otest)

    # getting results
    tp, tn, fp, fn, acc, f1 = get_results(score, test_label)
    print acc, f1

    # logging result in file
    print '\npool size : %d,\tnumber feature : %d,\t stopword : %d\n\tTrue positive : %d\n\tTrue negative : %d\n\tFalse positive : %d\n\tFalse negatie : %d\n\taccuracy : %f\n\tf1 score : %f\n'%(pool_size,num_feature,stp,tp, tn, fp, fn,acc, f1)
    open(res_dir+'res.txt','a').write('\npool size : %d,\tnumber feature : %d,\t stopword : %d\n\tTrue positive : %d\n\tTrue negative : %d\n\tFalse positive : %d\n\tFalse negatie : %d\n\taccuracy : %f\n\tf1 score : %f\n'%(pool_size,num_feature,stp,tp, tn, fp, fn,acc, f1))

if __name__ == "__main__":
    var_file = ['weights/model_variables.pickle']

    pool_size = 10
    num_feature = [1,0]
    stp = [1]
    parsing_type = ['dep']
    for i in range(len(var_file)):
        for nfeat in num_feature:
           test_fun(var_file[i], pool_size, nfeat, stp[i], parse_type=parsing_type[i])
