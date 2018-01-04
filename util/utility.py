
from practnlptools.tools import Annotator
import re, numpy as np
import warnings, random
warnings.filterwarnings("error")

def get_parents(words_data):
    '''
    return parent ids
    :param words_data: []; definded in line_wdata
    :return:
        p: []; parent list
    '''
    p=[]
    for i in words_data:
        p.append(words_data[i]['pid'])
    p = list(set(p))
    # if -1 in p:
    #     p.remove(-1)
    return p

def get_chunks(words_data):
    '''
    return chunk info
    :param words_data: []; definded in line_wdata
    :return:
        chk: [[wid1, wid2, ..] ...]; chunk info([]= wids in chunk)
    '''
    chk = []
    temp = []
    for i in sorted(words_data):
        if words_data[i]['chk'][0] == 'B' or words_data[i]['chk'][0] == 'I':
            temp.append(words_data[i]['wid'])
        elif words_data[i]['chk'][0] == 'E':
            temp.append(words_data[i]['wid'])
            chk.append(temp)
            temp = []
        elif words_data[i]['chk'][0] == 'S' or words_data[i]['chk'][0] == 'O':
            chk.append([i])
        else :
            print "Error Not a valid chunk Tag"
    return chk

def get_chunk_main(chks, dep_order):
    '''
    Get main word in chunks
    :param chks: [chunk1, chunk2, ...] chunk1=[wid1, wid2, ...]
    :param dep_order: [wid1, wid2, ...]
    :return:
        chk_main: [chunk_main1, chunk_main1, ...] respective to
    '''
    chk_main = []
    for chk in chks:
        if len(chk) == 1:
            chk_main.append(chk[0])
        else:
            # t_dep_order = []+dep_order
            top = -1
            for ch in chk:
                if ch in dep_order and top < dep_order.index(ch):
                    top = chk.index(ch)
            if top == -1:
                top=random.randrange(len(chk))
            chk_main.append(chk[top])
    return chk_main

def get_main_phrase(chunk, d1):
    mid = d1[0][0]
    clen = len(chunk)
    for i in range(clen):
        for j in chunk[i]:
            if j == mid:
                break
        if j == mid:
            break

    wp = [0 for l in chunk]
    ind=i-1
    while ind >= 0:
        wp[ind] = ind-i
        ind -= 1
    ind = i + 1
    while ind < clen:
        wp[ind] = ind - i
        ind += 1
    return wp

def get_dep(words_data):
    '''
    :return dependency in wids
    :param words_data: []; definded in line_wdata
    :param wid: if wid(word id) is given return all dependency tuple contain wid
    :return:
        d: [(child_wid, parent_wid), ...]; dependency between words
        or
        t_dep : [wid1, wid2, wid3, ...] dep. of wid
    '''
    d=[]
    for i in words_data:
        if words_data[i]['pid'] != -1:
            d.append((words_data[i]['wid'], words_data[i]['pid']))
    return d

def pdep_2_deporder_dep(p,d):
    '''
    return dependency order of parent wids and dependency dict
    :param p: [] parent list
    :param d: [] dependency
    :return:
        dep_order: [] ordered based on which compute first
        d1: {} wid depend on wids
    '''
    d1 = {}
    for i in p:
        d1[i] = []
        for j in d:
            if i == j[1]:
                d1[i].append(j[0])

    r = [0]
    dep_order = []
    if 0 not in d1:
        pass
    while r:
        node = r.pop(0)
        leaves = d1[node]
        for leave in leaves:
            if leave in p and leave not in dep_order:
                dep_order.append(leave)
                r.append(leave)
    dep_order = list(reversed(dep_order))
    return dep_order,d1

def get_order(d1, w_size):
    root = [0]
    order = []
    while root :
        t=[]
        for i in root:
            order += d1[i]
            t=[j for j in d1[i] if j in d1]
        root = t
    order = list(reversed(order))
    return order


def dep_2_hid_var(p, dep_order, d1, Word_ids):
    '''
    this function calculate different varable required for computation of rae
    :param p: parent wid list
    :param dep_order: dependency order of wid
    :param d1: as definded in pdep_2_deporder_dep
    :param Word_ids: word ids
    :return:
        h_index: hidden index
        h_vect: required for hidden vector calculation
        wp: required for hidden vector calculation

    explaination:
      h contain list of vector contain single vector combine
      h1 contain list of vector contain more then one vector
      h2 contain index of vector(in h2) to be combine to form tha ith vector in h_order vector( after visible vector form)
    eg
        Word_id = [1,2,3,4,5,6,7,8,9,10,11,12]
        dep_order = [3,7,8,5,11,12]
        h_order = [1,2,4,6,9,10,3,7,8,5,11,12]
        h = [1, 2, 4, 6, 9, 10]
        h1 = [[1,2,3], [6,7], [7,8], [3,4,5,8], [10,11], [5,9,11,12]]
        h11 = [3:[1,2,3], 7:[6,7], 8:[7,8], 5:[3,4,5,8], 11:[10,11], 12:[5,9,11,12]

        vects = [v1, v2, v3, v4 ,v5 ,v6 , v7, v8, v9, v10, v11, v12, h1 ,h2, h4, h6, h9, h10, h3, h7, h8, h5, h11, h12]
        h_index = {1:12,2:13,3:18,4:14,5:21,6:15,7:19,8:20,9:15,10:16,11:22,12:23}

        h_vect = [h1=[v1], h2=[v2], h4=[v4], h6=[v6], h9=[v9], h10=[v10], h3=[h1, h2, v3], h7=[h6, v7], h8=[h7, v8], h5=[h3, h4, v5, h8], h11=[h10, v11], h12=[h5,h9,h11,v12]]
        h_vect = [[0],[1],[3],[5],[10],[11],[12,13,2],[15,6],[19,7],[18,14,4,20],[17,10], [21,16,22,11]]
        ch_no = [[1],[1],[1],[1],[1],  [1], [1,1,1]  ,[1, 1],[2, 1],[3, 1, 1,3 ],[1 ,1 ], [8, 1, 2, 1 ]]

        vect_index = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,[0],[1],[4],[6],[9],[10],[12,13,2],[15,6],[19,7],[18,14,4,20],[17,10], [21,16,22,11]]

      h_order => h_index
      h_order order of calculation of hidden variable
      h_index means that ith value point to hi vector
    eg:
        h5 is at vector[h_index[5]]
      vect_index defind the index of vector required in calculation of vector(hidden_vector)
    '''
    w_size=len(Word_ids)
    h_order = []
    wp = []
    for i in Word_ids:
        if i not in p:
            h_order.append(i)
            wp.append([0.1])
    h_order += dep_order

    h11 = {key:sorted(value+[key]) for key,value in d1.items()}
    h11.pop(0)

    for i in dep_order:
        temp1 = [0 for j in h11[i]]
        ind1 = h11[i].index(i)
        for j in reversed(range(0,ind1)):
            temp1[j]=temp1[j+1]-1
        for j in range(ind1 + 1, len(temp1)):
            temp1[j] = temp1[j - 1] + 1
        wp.append(temp1)

    h_index = {}
    for i in range(len(h_order)):
        h_index[h_order[i]] = i + w_size
    h_vect = []
    hh_index = {}
    for i in range(len(Word_ids)):
        if Word_ids[i] not in p:
            h_vect.append([i])
    for i in dep_order:
        t = []
        for j in h11[i]:
            if j != i:
                t.append(h_index[j])
                hh_index[j]=h_index[i]
            else:
                t.append(Word_ids.index(i))
        h_vect.append(t)
    return h_index, h_vect, wp, hh_index

def get_child_no(h_vect, w_size):
    ch_no1=[]
    for i in range(len(h_vect)):
        if len(h_vect[i]) == 1:
            ch_no1.append([1.0])
        else:
            ch_no1.append([])
            for j in range(len(h_vect[i])):
                if h_vect[i][j] >= w_size:
                    ch_no1[i].append(float(sum(ch_no1[h_vect[i][j]-w_size])))
                else:
                    ch_no1[i].append(1.0)
    return ch_no1

def get_words_id(words_data):
    '''
    return list of words ids
    :param words_data:
    :return:
        wid: list of word ids
    '''
    wid = []
    for i in sorted(words_data):
        if type(i) is int:
            wid.append(words_data[i]['wid'])
    return list(set(wid))

def get_words_vect(words_data, Word_id, v_size):
    '''
    return list of words vectors
    :param words_data:
    :return:
        vect: list of words vectors
    '''
    vect = []
    for i in Word_id:
        vect.append(words_data[i]['vect'])
    return vect

def sigmoid(x):
    return np.power(1+np.exp(-x), -1)

def dsigmoid(x):
    t=sigmoid(x)
    return (1-t)*t

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return  1-np.square(np.tanh(x))

def tanh1(x):
    return np.tanh(x)/2+.5

def dtanh1(x):
    return  (1-np.square(np.tanh(x)))/2

def relu(x):
    return np.where(x<0,0,x)

def drelu(x):
    return np.where(x>0,1,0.01)

def preprocess(a):
    return np.power(1+np.exp(-a),-1)
#
# def preprocess(a):
#     return 2/(1+np.exp(-0.2*a))-1

# def preprocess(a):
#     return np.power(1+np.exp(-a),-1) - 0.5

# def preprocess(a):
#     return np.power(1+np.exp(-a),-1)

# def preprocess(a):
#     return np.tanh(a) / 2 + .5

def softmax(vect):
    x_exp = np.exp(vect)
    sum_exp = x_exp.sum()
    return x_exp/sum_exp
#
# def preprocess1(vect):
#     r,c=vect.shape
#     x_exp = np.exp(vect)
#     sum_exp = np.sum(x_exp,1).reshape((r,1))
#     return x_exp / sum_exp

def set_one(w_range, size1, size2):
    import numpy as np
    w = {}
    mean = 0
    sigma = 0.1
    if w_range is None:
        # return np.array([random.randint(0,1) for i in range(size1*size2)]).reshape((size1, size2))
        return np.ones((size1, size2))
    elif w_range%2 == 0:
        print "Input a proper weights range"
        exit()
    rng = int(w_range/2)
    a = range(-rng,rng+1)
    for i in a:
        w[i] = 0
    # w[0] = np.array([random.randint(0,1) for i in range(size1*size2)]).reshape((size1, size2))
    w[0] = np.ones((size1, size2))
    return w

def get_weight_matrices(w_range, size1, size2):
    import numpy as np
    w = {}
    mean = 0
    sigma = 0.1
    if w_range is None:
        return np.random.normal(mean, sigma, (size1, size2))
    elif w_range%2 == 0:
        print "Input a proper weights range"
        exit()
    rng = int(w_range/2)
    a = range(-rng,rng+1)
    for i in a:
        w[i] = 0
    w[0] = np.random.normal(mean, sigma, (size1, size2))
    return w

def get_zero_weight_matrices(w_range, size1, size2):
    import numpy as np
    w = {}
    if w_range is None:
        return np.zeros((size1, size2))
    elif w_range%2 == 0:
        print "Input a proper weights range"
        exit()
    rng = int(w_range/2)
    a = range(-rng,rng+1)
    for i in a:
        w[i] = 0
    w[0] = np.zeros((size1, size2))
    return w

def init_weight(size1, size2=0):
    mean = 0
    sigma = 0.1
    if size2 == 0:
        return np.random.normal(mean, sigma, (size1, 1))
    return np.random.normal(mean, sigma, (size1, size2))

def init_weight1(size1, size2=0):
    if size2 == 0:
        return np.random.uniform(0.0, 0.1, (size1, 1))
    return np.random.uniform(0.0, 0.1, (size1, size2))

def zero_weight(size1, size2=0):
    if size2 == 0.0:
        return np.zeros((size1, 1))
    return np.zeros((size1, size2))

def init_one(size1, size2):
    # return np.array([random.randint(0,1) for i in range(size1*size2)]).reshape((size1, size2))
    return np.ones((size1, size2))

def save2pickle(w,name):
    import pickle
    with open(name, 'wb') as handle:
        pickle.dump(w, handle, protocol=pickle.HIGHEST_PROTOCOL)

def add_ids(words_data, wid, pid):
    if wid in words_data:
        if words_data[wid]['pid'] != pid:
            words_data[wid+.1]={'wid':wid,'pid':pid}
    else:
        words_data[wid]={'wid':wid,'pid':pid}
    return

# from dep_correction import depth

def extract_feature_using_senna(line):
    '''
    Takes line in and data out
    :param line: an english sentence
    :return:
        data : [word, number, pos, chunk_info, ner_info, parent_number]
    '''
    annotator = Annotator()
    feature = annotator.getAnnotations(line, dep_parse=True)

    if feature['dep_parse']=='':
        return -2
    a = feature['dep_parse'].split('\n')
    words_data = {}
    d=[]
    for i in a:
        dep = re.sub(r'^[^\(]+\(|\)$', '', i)
        try:
            p,c = dep.split(', ')
        except ValueError:
            pass
        try:
            t1 = p.split('-')
            pid = int(t1[len(t1)-1])
            t2 = c.split('-')
            wid = int(t2[len(t2)-1])
        except ValueError:
            if re.match('[\d]+\'',t1[len(t1)-1]):
                pid = int(re.sub(r'\'','',t1[len(t1)-1]))+0.1
                t2 = c.split('-')
                wid = int(t2[len(t2) - 1])
            elif re.match('[\d]+\'',t2[len(t2)-1]):
                pass
            continue
        d.append((wid,pid))
    t1 = [id for id in d]
    d,_=remove_dep(t1)
    for wid, pid in d:
        add_ids(words_data,wid,pid)
    for i in range(len(feature['words'])):
        if i+1 not in words_data:
            words_data[i+1] = {'wid':i+1, 'pid':-1, 'word':feature['words'][i], 'chk': feature['chunk'][i][1], 'ner':feature['ner'][i][1], 'pos':feature['pos'][i][1]}
        elif i+1 in words_data:
            words_data[i + 1]['word'] = feature['words'][i]
            words_data[i + 1]['chk'] = feature['chunk'][i][1]
            words_data[i + 1]['ner'] = feature['ner'][i][1]
            words_data[i + 1]['pos'] = feature['pos'][i][1]
    words_data['syntax_tree'] = feature['syntax_tree']
    words_data['tree'] = feature['tree']
    words_data['verbs'] = feature['verbs']
    words_data['srl'] = feature['srl']
    # Global.accepted += 1
    return words_data

def extract_wpcn_feature_using_senna(line):
    '''
    Takes line in and data out
    :param line: an english sentence
    :return:
        data : [word, number, pos, chunk_info, ner_info, parent_number]
    '''
    annotator = Annotator()
    feature = annotator.getAnnotations(line, dep_parse=False)
    words_data = {}
    for i in range(len(feature['words'])):
        words_data[i+1] = {'wid':i+1, 'word':feature['words'][i], 'chk': feature['chunk'][i][1], 'ner':feature['ner'][i][1], 'pos':feature['pos'][i][1]}
    words_data['syntax_tree'] = feature['syntax_tree']
    words_data['tree'] = feature['tree']
    words_data['verbs'] = feature['verbs']
    words_data['srl'] = feature['srl']
    return words_data

def extract_batchfeature_using_senna(lines):
    '''
    Takes line in and data out
    :param line: an english sentence
    :return:
        data : [word, number, pos, chunk_info, ner_info, parent_number]
    '''
    annotator = Annotator()
    features = annotator.getBatchAnnotations(lines, dep_parse=True)
    words_datas = []
    for feature in features:
        if feature['dep_parse']=='':
            continue
        a = feature['dep_parse'].split('\n')
        words_data = {}
        d = []
        for i in a:
            # wid=None;pid=None
            dep = re.sub(r'^[^\(]+\(|\)$', '', i)
            try:
                p, c = dep.split(', ')
            except ValueError:
                pass
            try:
                t1 = p.split('-')
                pid = int(t1[len(t1) - 1])
                t2 = c.split('-')
                wid = int(t2[len(t2) - 1])
            except ValueError:
                if re.match('[\d]+\'', t1[len(t1) - 1]):
                    pid = int(re.sub(r'\'', '', t1[len(t1) - 1])) + 0.1
                    t2 = c.split('-')
                    wid = int(t2[len(t2) - 1])
                elif re.match('[\d]+\'', t2[len(t2) - 1]):
                    pass
                continue
            d.append((wid, pid))
        t1 = [id for id in d]
        d, _ = remove_dep(t1)
        for wid, pid in d:
            add_ids(words_data, wid, pid)
        for i in range(len(feature['words'])):
            if i + 1 not in words_data:
                words_data[i + 1] = {'wid': i + 1, 'pid': -1, 'word': feature['words'][i],
                                     'chk': feature['chunk'][i][1], 'ner': feature['ner'][i][1],
                                     'pos': feature['pos'][i][1]}
            elif i + 1 in words_data:
                words_data[i + 1]['word'] = feature['words'][i]
                words_data[i + 1]['chk'] = feature['chunk'][i][1]
                words_data[i + 1]['ner'] = feature['ner'][i][1]
                words_data[i + 1]['pos'] = feature['pos'][i][1]
        # Global.accepted += 1
        words_data['syntax_tree'] = feature['syntax_tree']
        words_data['tree'] = feature['tree']
        words_data['verbs'] = feature['verbs']
        words_data['srl'] = feature['srl']
        words_datas.append(words_data)
    return words_datas

def remove_dep(edges, root=0):
    edges = list(set(edges))
    te=len(edges)
    t1=[]
    p=[root]
    p_all=[root]
    while len(p) > 0:
        c=[]
        for i in p:
            for ci,pi in edges:
                if pi == i and ci not in p_all:
                    c.append(ci)
        for i in p:
            for j in c:
                if (j,i) in edges and j not in p_all:
                    t1.append((j,i))
                    edges.remove((j,i))
        p=c
        p_all+=c
        if len(p_all)>te*2:
            raise MemoryError
    return t1, edges
def cleanwd(wds):
    rmwd = []
    for i in wds:
        if wds[i]['pid']==-1:
            rmwd.append(i)
    for j in rmwd:
        wds.pop(j)
    return wds

def get_parent_detail(Word_ids, dep_order, h_index, hh_index, h_vect, wp, p):
    w_size = len(Word_ids)
    vnodes = [(i, Word_ids[i]) for i in range(w_size)]
    t1 = [i for i in Word_ids if i not in p] + dep_order[:-1]
    hnodes = [(i + w_size, t1[i]) for i in range(len(t1))]
    # nodes = dict(vnodes + hnodes)
    # print "hnode        :", hnodes
    # print "nodes        :", nodes
    # print "h_index      :", h_index
    # print "hh_index     :", hh_index
    # print "h_vect       :", h_vect
    # try:
    n_parent = dict([(i[0], h_index[i[1]]) for i in vnodes] + [(i[0], hh_index[i[1]]) for i in hnodes])
    # except KeyError:
    #     pass
    # print "n_parent     :", n_parent

    npa_sibling = {}
    nonc = [k for k, v in n_parent.items() if v == w_size * 2 - 1]
    for i in n_parent:
        if i not in nonc:
            npa_parent = n_parent[n_parent[i]]
            t1 = h_vect[npa_parent - w_size]
            npa_sibling[i] = [j for j in t1 if j != n_parent[i]]
        else:
            npa_sibling[i] = []
    # print "npa_sibling  :", npa_sibling

    self_wp = {i: 0 for i in range(w_size)}
    for i in range(w_size, 2 * w_size - 1):
        t1 = h_vect[n_parent[i] - w_size]
        n_index = t1.index(i)
        self_wp[i] = wp[n_parent[i] - w_size][n_index]
    # print "self_wp      :", self_wp

    npa_wp = {}
    for i in n_parent:
        if i not in nonc:
            npa_wp[i] = self_wp[n_parent[i]]
        else:
            npa_wp[i] = np.nan
    # print "npa_wp       :", npa_wp
    npas_wp = {}
    for i in n_parent:
        if i not in nonc:
            npas_wp[i] = [self_wp[j] for j in npa_sibling[i]]
        else:
            npas_wp[i] = [np.nan]
    # print "npas_wp      :", npas_wp
    return n_parent, npa_sibling, self_wp, npa_wp, npas_wp

if __name__=='__main__':
    # Global.init()
    # sent1 = 'contrary to court decision firmly believe congress gave f.t.c. authority to implement national do not call list congressmen said in statement'
    wd={1: {'wid': 1, 'word': 'gyorgy', 'chk': 'B-NP', 'pid': 3, 'pos': 'FW', 'ner': 'O'}, 2: {'wid': 2, 'word': 'heizler', 'chk': 'I-NP', 'pid': 3, 'pos': 'FW', 'ner': 'O'}, 3: {'wid': 3, 'word': 'head', 'chk': 'E-NP', 'pid': 8, 'pos': 'NN', 'ner': 'O'}, 4: {'wid': 4, 'word': 'of', 'chk': 'S-PP', 'pid': -1, 'pos': 'IN', 'ner': 'O'}, 5: {'wid': 5, 'word': 'local', 'chk': 'B-NP', 'pid': 7, 'pos': 'JJ', 'ner': 'O'}, 6: {'wid': 6, 'word': 'disaster', 'chk': 'I-NP', 'pid': 7, 'pos': 'NN', 'ner': 'O'}, 7: {'wid': 7, 'word': 'unit', 'chk': 'E-NP', 'pid': 3, 'pos': 'NN', 'ner': 'O'}, 8: {'wid': 8, 'word': 'said', 'chk': 'S-VP', 'pid': 0, 'pos': 'VBD', 'ner': 'O'}, 9: {'wid': 9, 'word': 'coach', 'chk': 'S-NP', 'pid': 10, 'pos': 'NN', 'ner': 'O'}, 10: {'wid': 10, 'word': 'carrying', 'chk': 'S-VP', 'pid': 8, 'pos': 'VBG', 'ner': 'O'}, 11: {'wid': 11, 'word': '38', 'chk': 'B-NP', 'pid': 12, 'pos': 'CD', 'ner': 'O'}, 12: {'wid': 12, 'word': 'passengers', 'chk': 'E-NP', 'pid': 10, 'pos': 'NNS', 'ner': 'O'}}
    print wd
    d = get_dep(wd)
    wd=cleanwd(wd)
    Word_ids = get_words_id(wd)
    w_size = len(Word_ids)
    p = get_parents(wd)
    Word_ids = get_words_id(wd)
    w_size = len(Word_ids)
    dep_order, d1 = pdep_2_deporder_dep(p, d)
    # h_index, h_vect1, wp1 , hh_index = dep_2_hid_var(p, dep_order, d1, Word_ids)

    h_index = []
    chunk = get_chunks(wd)
    chunk_main= get_chunk_main(chks=chunk, dep_order=dep_order)
    h_vect = []
    wp = []
    for i in range(len(chunk)):
        temp=[]
        # temp1=[]
        # if len(chunk[i]) > 1:
        for j in chunk[i]:
            temp.append(Word_ids.index(j))
        h_vect.append(temp)

        temp1 = [0 for l in chunk[i]]
        mid = chunk[i].index(chunk_main[i])
        ite = mid-1
        while ite>=0:
            temp1[ite] = ite - mid
            ite-=1
        ite = mid+1
        while ite < len(chunk[i]):
            temp1[ite] = ite - mid
            ite += 1
        wp.append(temp1)
    print d1
    print h_index
    print h_vect
    print