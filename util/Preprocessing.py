import re, random

class preprocess:

    def __init__(self, parsing_type, structure_type, stopword, wvect):

        self.NHIERARCHY = 0
        self.HIERARCHY = 1
        if parsing_type == 'chk':
            self.processing = self.chk_processing
        elif parsing_type == 'syn':
            self.processing = self.syn_processing
        elif parsing_type == 'dep':
            self.processing = self.dep_processing

        if structure_type == 'nh':
            self.struct = self.NHIERARCHY
        elif structure_type == 'h':
            self.struct = self.HIERARCHY

        self.wd_vect = wvect
        self.stp = stopword

    def process_words_data(self, wds):
        wd_extra = []
        wpresent = []
        counts = 0
        stopwds = [i.strip('\n') for i in open('/media/zero/41FF48D81730BD9B/DT_RAE/Chunk_rnn/config/stopword.txt', 'r')]
        for iword in range(len(wds)):

            if wds[iword] == -2 and not wds[iword]:
                continue

            if self.stp == 0:
                wd = {}; oth={}
                for i in wds[iword]:
                    if type(i) == int:
                        wd[i] = wds[iword][i]
                    else:
                        oth[i] = wds[iword][i]
                wds[iword] = self.removing_stopword(wd, stopwds)
                wds[iword].update(oth)

            h_index, h_vect, wp = self.processing(wds[iword])
            t=[]
            for i in wp:
                t+=[j for j in i]
            wpresent = list(set(wpresent+t))

            words, vects, count = {}, {}, 0
            for i in wds[iword]:
                if type(i) == int:
                    words[count] = wds[iword][i]['word']
                    vects[count] = self.wd_vect.get_word_vect(wds[iword][i]['word'].lower())
                    count += 1
            wd_extra.append({'w_size': len(words), 'h_vect': h_vect, 'words':words, 'vects': vects, 'wp': wp})
            counts += 1
            #if counts%10000 == 0:
             #   print "%d/%d are processed ...." % (len(wd_extra), len(wds))
        #print "%d/%d are processed ...."%(len(wd_extra), len(wds))
        #print "Count of new vectored words :", self.wd_vect.nov_count
        return wd_extra, wpresent

    def chk_processing(self, word_data):
        wd = {}
        for i in word_data:
            if type(i) == int:
                wd[i] = word_data[i]
        p = []; dep = []
        for i in wd:
            p.append(wd[i]['pid'])
            dep.append((wd[i]['wid'], wd[i]['pid']))
        p = sorted(list(set(p)))
        chunk = get_chunks(wd)
        dep_order, d1 = pdep_2_deporder_dep(p, dep)
        chunk_main = get_chunk_main(chunk, dep_order)

        chunk_dep = {i[0]: i[1] for i in dep}
        temp_dep = []
        for i in dep:
            if i[0] in chunk_main + [0] and i[1] in chunk_main + [0]:
                temp_dep.append(i)

        # correcting 0 as parant not present problem
        # #########################################################################################
        a = []
        for i in temp_dep:
            a += list(i)
        a = list(set(a))
        if 0 not in a:
            a.append(0)
            temp = {i[1]: i[0] for i in dep}
            t1 = 0
            for j in range(len(dep)):
                if t1 not in temp:
                    break
                t1 = temp[t1]
                if t1 in chunk_main + [0]:
                    temp_dep.append((t1, 0))
                    break
        ###########################################################################################

        # correcting transitive chunk_main parent dependency problem
        ###########################################################################################
        a = []
        for i in temp_dep:
            a += list(i)
        a = list(set(a))
        t = chunk_main + [0]
        if len(set(a)) != len(t):
            # print
            for i in chunk_main:
                if i not in a:
                    t1 = i
                    for j in range(len(dep)):
                        if t1 not in chunk_dep:
                            break
                        t1 = chunk_dep[t1]
                        if t1 in chunk_main + [0]:
                            temp_dep.append((i, t1))
                            break
        ############################################################################################

        # correcting no parant problem
        ###########################################################################################
        a = []
        for i in temp_dep:
            a += list(i)
        a = list(set(a))
        if len(a) != len(t):
            a1 = []
            for i in t:
                if i not in a:
                    a1.append(i)
            top = d = a1.pop(0)
            for i in a1:
                temp_dep.append((i, d))
                d = i
            diff = 1000
            c = -1
            for i in a:
                if abs(top - i) < diff:
                    diff = abs(top - i)
                    c = i
            temp_dep.append((top, c))
        ###########################################################################################

        Word_ids = []
        for i in chunk:
            Word_ids += i
        Word_ids.sort()
        h_vect = [[Word_ids.index(j) for j in i] for i in chunk]
        wp = []
        for i in range(len(chunk)):
            if len(chunk[i]) == 1:
                wp.append([0.1])
            else:
                temp1 = [0 for l in chunk[i]]
                mid = chunk[i].index(chunk_main[i])
                ite = mid - 1
                while ite >= 0:
                    temp1[ite] = ite - mid
                    ite -= 1
                ite = mid + 1
                while ite < len(chunk[i]):
                    temp1[ite] = ite - mid
                    ite += 1
                wp.append(temp1)
        words = {i: wd[i]['word'] for i in wd}
        return None, h_vect, wp

    def syn_processing(self, wrd):
        wd = {}
        for i in wrd:
            if type(i) == int:
                wd[i] = wrd[i]

        t = re.sub(r'([\)\(])', r' \1 ', wrd['tree'])
        t = re.sub(r'[ ]{2,}', r' ', t)
        t = t.split(' ')
        p = []
        id = 1
        ind = 0
        h_vect = []
        h_index = {}

        for i in range(len(t)):
            if t[i] != '(' and t[i] != ')' and t[i] != '' and t[i] == wrd[id]['word']:
                temp = id
                h_index[temp] = ind
                id += 1
                ind += 1
                t[i] = temp
                p.append(wrd[temp]['pid'])
            else:
                p.append(t[i])

        while i < len(p):
            if p[i] == -1:
                p.pop(i)
                t.pop(i)
            i += 1

        # check for redudent brackets
        bs = [];
        be = []
        i = 0
        while i < len(t):
            if t[i] == '(':
                bs.append(i)
            elif t[i] == ')':
                be.append(i)
            if len(be) == 2:
                if bs[-1] - bs[-2] == 1 and be[1] - be[0] == 1:
                    t.pop(be[0])
                    t.pop(bs[-1])
                    be[1] -= 2
                    bs[-1] -= 1
                    i -= 2
                a = be.pop(0)
                for j in sorted(range(len(bs)), reverse=1):
                    if bs[j] < a:
                        bs.pop(j)
                        break
            i += 1
        stack = []
        i = 0
        h_ind = 0
        while len(t) > 1:
            if t[i] is '':
                t.pop(i)
                continue
            elif t[i] is '(':
                stack.append(i)
            elif t[i] is ')':
                h_vect.append([])
                for j in range(stack[-1] + 1, i):
                    h_vect[h_ind].append(h_index[t[j]])
                h_index[id] = ind
                for l in range(stack[-1], i):
                    t.pop(stack[-1])
                t[stack[-1]] = id
                i = stack[-1]
                stack.pop()
                if not h_vect[h_ind]:
                    i += 1
                    continue
                h_ind += 1
                ind += 1
                id += 1
            i += 1

        wp = []
        for i in range(len(h_vect)):
            wp.append([])
            count = len(h_vect[i]) - 1
            for j in h_vect[i]:
                wp[i].append(count)
                count -= 1
        return h_index, h_vect, wp

    def dep_processing(self, word_data):
        wd = {}
        for i in word_data:
            if type(i) == int:
                wd[i] = word_data[i]
        p=[];d=[];Word_ids=[]
        for i in sorted(wd):
            Word_ids.append(wd[i]['wid'])
            p.append(wd[i]['pid'])
            d.append((wd[i]['wid'],wd[i]['pid']))
        p=sorted(list(set(p)))
        dep_order, d1 = pdep_2_deporder_dep(p, d)
        h_index, h_vect, wp , hh_index = dep_2_hid_var(p, dep_order, d1, Word_ids)
        return h_index, h_vect, wp

    def removing_stopword(self, wd, stopwds):
        wd_index = [i for i in wd]
        p=[]
        for i in sorted(wd):
            p.append(wd[i]['pid'])
        p=sorted(list(set(p)))
        for i in wd_index:
            if i not in p and wd[i]['word'] in stopwds:
                wd.pop(i)
        return wd


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
    w_size = len(Word_ids)
    h_order = []
    wp = []
    for i in Word_ids:
        if i not in p:
            h_order.append(i)
            wp.append([0.1])
    h_order += dep_order

    h11 = {key: sorted(value + [key]) for key, value in d1.items()}
    h11.pop(0)

    for i in dep_order:
        temp1 = [0 for j in h11[i]]
        ind1 = h11[i].index(i)
        for j in reversed(range(0, ind1)):
            temp1[j] = temp1[j + 1] - 1
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
                hh_index[j] = h_index[i]
            else:
                t.append(Word_ids.index(i))
        h_vect.append(t)
    return h_index, h_vect, wp, hh_index


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
                top = random.randrange(len(chk))
            chk_main.append(chk[top])
    return chk_main
