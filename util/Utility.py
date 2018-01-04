import numpy as np, re, pickle
from random import shuffle

def init_weight(size1, size2=0, mean = 0, sigma = 0.1):
    if size2 == 0:
        return np.random.normal(mean, sigma, (size1, 1))
    return np.random.normal(mean, sigma, (size1, size2))


def get_n_feature(line1, line2):
    nfeat = [0,0,0]
    p = re.compile(' [0-9]+ | [0-9]+\.[0-9]+ ')
    m1 = p.findall(line1)
    m2 = p.findall(line2)

    if m1 and m2:
        nfeat[0] = 0
    elif not m1 and not m2:
        nfeat[0]=1
        return nfeat
    else:
        return nfeat

    if len(m1) == len(m2):
        nfeat[0] = 1
        tm1 = [i for i in m1]
        tm2 = [i for i in m2]
        for i in m1:
            if i in tm2:
                tm2.remove(i)
                tm1.pop(0)
        if not tm2 and not tm1:
            nfeat[1]=1
    else:
        nfeat[0] = 0
        nfeat[0] = 0
        tm = [m1, m2] if len(m1)<len(m2) else [m2, m1]
        for i in tm[1]:
            if i in tm[0]:
                tm[0].remove(i)
        if not tm[0]:
            nfeat[2] = 1
    return nfeat

def dynamic_pooling(in_matrix, pool_size, pf='min'):

    if pf == 'max':
        pool_fun = np.max
    elif pf == 'mean':
        pool_fun = np.mean
    else:
        pool_fun = np.min

    output_matrix = np.zeros((pool_size,pool_size))
    dim1, dim2 = in_matrix.shape
    while dim1 < pool_size:
        in_matrix = np.concatenate([[in_matrix[int(i / 2), :]] for i in range(dim1 * 2)], axis=0)
        dim1,_ = in_matrix.shape
    while dim2 < pool_size:
        in_matrix = np.concatenate([[in_matrix[:,int(i/2)].tolist()] for i in range(dim2*2)],axis=0).transpose()
        _,dim2 = in_matrix.shape

    # quot1 = floor(dim1 / pool_size);
    # quot2 = floor(dim2 / pool_size);
    qout1 = int(np.floor(dim1 / pool_size))
    qout2 = int(np.floor(dim2 / pool_size))

    # rem1 = dim1 - quot1 * pool_size;
    # rem2 = dim2 - quot2 * pool_size;
    rem1 = dim1 - qout1 * pool_size
    rem2 = dim2 - qout2 * pool_size

    # vec1 = [0;cumsum(quot1 * ones(pool_size, 1) + [zeros(pool_size - rem1, 1); ones(rem1, 1)])];
    # vec2 = [0;cumsum(quot2 * ones(pool_size, 1) + [zeros(pool_size - rem2, 1);ones(rem2, 1)])];
    t11 = qout1 * np.ones((pool_size, 1))
    t12 = np.concatenate((np.zeros((pool_size - rem1, 1)), np.ones((rem1, 1))))
    t21 = qout2 * np.ones((pool_size, 1))
    t22 = np.concatenate((np.zeros((pool_size - rem2, 1)), np.ones((rem2, 1))))
    vec1 = np.concatenate(([[0]] , np.cumsum(t11+t12,axis=0)),axis=0,)
    vec2 = np.concatenate(([[0]] , np.cumsum(t21+t22,axis=0)),axis=0)

    # pos = zeros(size(output_matrix));
    # pos = cat(3, pos, pos);
    # if method == 1
    #     func = @mean;
    #     elseif
    #     method == 2
    #     func = @min;
    #     end

    #for i=1:pool_size
    # for j=1:pool_size
    #    pooled = input_matrix(vec1(i) + 1:vec1(i + 1), vec2(j) + 1:vec2(j + 1));
    #    output_matrix(i, j) = func(pooled(:));
    #   end
    # end
    # disp(output_matrix);

    for i in range(pool_size):
        for j in range(pool_size):
            l11=int(vec1[i]); l12=int(vec1[i + 1])
            l21=int(vec2[j]); l22=int(vec2[j + 1])
            pooled = in_matrix[l11:l12, l21:l22]
            output_matrix[i,j] = pool_fun(pooled)
    return output_matrix

def similarity_matrix(x1, x2):
    s_matrix = np.zeros((len(x1), len(x2)))
    for i in x1:
        for j in x2:
            s_matrix[i, j] = np.linalg.norm(x1[i]-x2[j])
    s_min = {}
    for i in x1:
        # s_min[(x1[i], x2[np.argmin(s_matrix[i,])])] = np.amin(s_matrix[i,])
        s_min[i] = np.amin(s_matrix[i,])
    return s_min, s_matrix

def mini_batch(data, data_size, batch_size):
    batches = []
    shuffle(data)
    i=0
    for i in range(1,data_size/batch_size):
        batches.append(data[(i-1)*batch_size: i*batch_size])
    if data_size%batch_size != 0:
        batches.append(data[i*batch_size: data_size])
    return batches

def unzip(data):
    x=[]
    y=[]
    for i in data:
        x.append(i[0])
        y.append(i[1])
    return x, y

def get_results(score, y_test):
    tp = 0.0;    fp = 0.0;    fn = 0.0;    tn = 0.0;
    for i in range(len(y_test)):
        # fd.write("\ndesire score : " + str(y_test[i]) + " obtained : " + str(score[i]) + " sentences : " + sents[i] + '\n')
        if y_test[i] == 1:
            if score[i] == 1:
                tp += 1
            else:
                fn += 1
        elif y_test[i] == 0:
            if score[i] == 1:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + fp + tn + fn)
    return tp, tn, fp, fn, acc, f1

def get_msrp_data(stp):
    train_set = pickle.load(open("./MSRP/train/msr_paraphrase_train"+str(stp)+".pickle",'rb'))
    train_label = pickle.load(open("./MSRP/train/msr_paraphrase_trainscore"+str(stp)+".pickle",'rb'))
    train_sent = pickle.load(open("./MSRP/train/msr_paraphrase_trainsent"+str(stp)+".pickle",'rb'))
    test_set = pickle.load(open("./MSRP/test/msr_paraphrase_test"+str(stp)+".pickle",'rb'))
    test_label = pickle.load(open("./MSRP/test/msr_paraphrase_testscore"+str(stp)+".pickle",'rb'))
    test_sent = pickle.load(open("./MSRP/test/msr_paraphrase_testsent"+str(stp)+".pickle",'rb'))
    return train_set, train_label, test_set, test_label
    # return train_set[:100], train_label[:50], train_set[:100], train_label[:50]
