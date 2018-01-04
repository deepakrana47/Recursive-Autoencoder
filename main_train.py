import pickle
from models.recursive_autoencoder import stack_RAE
from util.optimization import Optimization, Optimization_variable
from util.Preprocessing import preprocess
from util.Activations import *
from util.Vectorization import Word_vector
import os

# def usage():
#     print "Usage : python RAE_trainning.py [options]"
#     print "options:"
#     print "\t -dir for pre defined files contain pickle data for trainning"
#     print "\t -in infile: infile has location of pickle file for trainning"
#     print "\t -neta value: learning rate(default=0.001)"
#     print "\t -hlayer value: Number of nodes in hidden layer"
#     print "\t -insize value: word vector size(50 or 200)"
#     print "\t -regu value: reguleraziation (default=0.01)"
#     print "\t -parse-tree: chk/syn/dep"
#     print "\t -type: h(hierarcal)/nh(non-hierarcal)"
#     print "\t -mfile model_file_name: filename to save trained model"
#     print "\t -wload weight_file_name: filename contain save trained model"
#     print '\t -batch_size iterations'
#     print '\t -stopwrd include stopword(by default 0)'
#     print '\t -epoch epoch size\n\n'
#     print '\t -method optimization method (default=rmsprop)\n\n'

def print_setting(args):
    print "\n\nTrainning Parameter:\n neta : %f \n word vector size :%d \n pharse vector size :%d \n batch size :%d \n optimization method :%s \n model :%s \n Stopword :%d \n parse-type :%s \n type :%s \n epoch :%d \n Input flag :%s \n Input file :%s \n loaded model file :%s \n save model file :%s"% (args['neta'], args['v_size'], args['h_size'], args['batch_size'], args['method'],args['model'], args['stp'], args['parse-type'], args['type'], args['epoch'], args['flag'], args['src'], args['wload'], args['wfname'])

def log_data(args):
    wt = args['wfname'].split('/')[-1]
    ddir = args['wfname']
    if not os.path.isdir(ddir):
        os.mkdir(ddir)
    open(ddir + '/setting.txt', 'w').write("\n\nTrainning Parameter:\n neta : %f \n word vector size :%d \n pharse vector size :%d \n batch size :%d \n optimization method :%s \n model :%s \n Stopword :%d \n parse-type :%s \n type :%s \n epoch :%d \n Input flag :%s \n Input file :%s \n loaded model file :%s \n save model file :%s"% (args['neta'], args['v_size'], args['h_size'], args['batch_size'], args['method'],args['model'], args['stp'], args['parse-type'], args['type'], args['epoch'], args['flag'], args['src'], args['wload'], args['wfname']))
    wfname = ddir + "/model_variables.pickle"
    log = {}
    log['iter'] = ddir + '/iter_count.pickle'
    log['epoch'] = ddir + '/epoch.pickle'
    log['iter_err'] = ddir + '/iter_err_count.pickle'
    log['rae'] = ddir + '/log.txt'
    return wfname, log

def create_model(model_type, op_method, i_size, o_size, neta, wpresent, logg, vector):
    a = None
    if model_type == 'RNN':
        pass
    else:
        opt_var = Optimization_variable(method=op_method, i_size=i_size, o_size=o_size, model_type=model_type, option={'wpresent':wpresent})
        opt = Optimization(optimization_variable=opt_var, method=op_method, learning_rate=neta)
        a = stack_RAE(input_size=i_size, hidden_size=o_size, optimization=opt,hidden_activation=args['hactiv'][0],hidden_deactivation=args['hactiv'][1],output_activation=args['oactiv'][0],output_deactivation=args['oactiv'][1], wpresent=wpresent, log_file=logg, log=1, vector=vector)
    return a

def main_train(args):

    print_setting(args)
    a=raw_input("Wanted to exit press q:")
    exit() if a == 'q'or a=='Q' else None

    wfname, log = log_data(args)

    # words_data = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/110.pickle", 'rb'))
    words_data = pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/msr_paraphrase_train.pickle', 'rb')) \
                 + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/1all-news.pickle', 'rb')) \
                 + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/2all-news.pickle', 'rb')) \
                 + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/3all-news.pickle', 'rb')) \
                 + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/4all-news.pickle', 'rb')) \
                 + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/5all-news.pickle', 'rb'))

    wvect = Word_vector(args['v_size'])
    data_processing = preprocess(parsing_type=args['parse-type'], structure_type=args['type'], stopword=args['stp'], wvect=wvect)
    data, wpresent = data_processing.process_words_data(words_data)
    nn = create_model(args['model'], args['method'], args['v_size'], args['h_size'], neta=args['neta'], wpresent=wpresent, logg = log['rae'], vector=wvect)
    if args['wload']:
        nn.load_variables(args['wload'])
    nn.train(xs=data, epoch=args['epoch'], batch_size=args['batch_size'])
    nn.save_variables(wfname)
    print "model variables saved."
    wvect.save_vector('/'.join(wfname.split('/')[:-1])+'/vectors.pickle')
    print "vector variables saved."
    return

if __name__ == '__main__':

    args = {
            'neta': 0.001,
            'v_size': 50,
            'h_size': 100,
            'batch_size': 100,
            'method': 'rmsprop',
            'model': 'RAE',
            'stp': 0,
            'parse-type': 'dep',
            'type': 'h',
            'epoch': 5,
            'hactiv':[elu, delu],
            'oactiv':[tanh, dtanh],
            'flag': 'd',
            'src': '',
            'wload': '',
            'wfname':'/media/zero/41FF48D81730BD9B/DT_RAE/IMPLEMENTATION/weights/2' ,
            }
    n = "%s_%d_%d_%d_%s_%s_%s_%s_%s"%(args['model'],args['v_size'],args['h_size'],args['stp'],args['method'],args['parse-type'],args['type'],args['hactiv'][0].__name__,args['oactiv'][0].__name__)
    args['wfname'] += n
    main_train(args)
    exit()