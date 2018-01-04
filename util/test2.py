import pickle, re
from utility import extract_feature_using_senna
# wrd = extract_feature_using_senna('She had been critically ill since having surgery at Baptist Hospital on May 7 to replace a heart valve')
wrd = extract_feature_using_senna('U.N. inspectors found traces of highly enriched weapons-grade uranium at an Iranian nuclear facility, a report by the U.N. nuclear agency says')
# wrd = extract_feature_using_senna('United Nations inspectors have discovered traces of highly enriched uranium near an Iranian nuclear facility heightening worries that the country may have a secret nuclear weapons program')
# wrd = extract_feature_using_senna('United Nations inspectors have discovered traces of highly enriched uranium near an Iranian nuclear facility, heightening worries that the country may have a secret nuclear weapons program')
# print wrd


# def syn_processing(wrd):
#     wd = {}
#     for i in wrd:
#         if type(i) == int:
#             wd[i] = wrd[i]
#
#     t = re.sub(r'([\)\(])', r' \1 ', wrd['tree'])
#     t = re.sub(r'[ ]{2,}', r' ', t)
#     t = t.split(' ')
#     p = []
#     id = 1
#     ind = 0
#     h_vect = []
#     h_index = {}
#
#     for i in range(len(t)):
#         if t[i] != '(' and t[i] != ')' and t[i] != '' and t[i] == wrd[id]['word']:
#             temp = id
#             h_index[temp] = ind
#             id += 1
#             ind += 1
#             t[i] = temp
#             p.append(wrd[temp]['pid'])
#         else:
#             p.append(t[i])
#
#     while i < len(p):
#         if p[i] == -1:
#             p.pop(i)
#             t.pop(i)
#         i += 1
#
#     # check for redudent brackets
#     bs = [];
#     be = []
#     i = 0
#     while i < len(t):
#         if t[i] == '(':
#             bs.append(i)
#         elif t[i] == ')':
#             be.append(i)
#         if len(be) == 2:
#             if bs[-1] - bs[-2] == 1 and be[1] - be[0] == 1:
#                 t.pop(be[0])
#                 t.pop(bs[-1])
#                 be[1] -= 2
#                 bs[-1] -= 1
#                 i -= 2
#             a = be.pop(0)
#             for j in sorted(range(len(bs)), reverse=1):
#                 if bs[j] < a:
#                     bs.pop(j)
#                     break
#         i += 1
#     stack = []
#     i = 0
#     h_ind = 0
#     while len(t) > 1:
#         if t[i] is '':
#             t.pop(i)
#             continue
#         elif t[i] is '(':
#             stack.append(i)
#         elif t[i] is ')':
#             h_vect.append([])
#             for j in range(stack[-1] + 1, i):
#                 h_vect[h_ind].append(h_index[t[j]])
#             h_index[id] = ind
#             for l in range(stack[-1], i):
#                 t.pop(stack[-1])
#             t[stack[-1]] = id
#             i = stack[-1]
#             stack.pop()
#             if not h_vect[h_ind]:
#                 i += 1
#                 continue
#             h_ind += 1
#             ind += 1
#             id += 1
#         i += 1
#
#     wp = []
#     for i in range(len(h_vect)):
#         wp.append([])
#         count = len(h_vect[i]) - 1
#         for j in h_vect[i]:
#             wp[i].append(count)
#             count -= 1
#     return h_index, h_vect, wp
#
# _, h_vect, wp = syn_processing(wrd)
# print h_vect
# print wp


def removing_stopword( wd, stopwds):
    wd_index = [i for i in wd]
    p = []
    for i in sorted(wd):
        p.append(wd[i]['pid'])
    p = sorted(list(set(p)))
    for i in wd_index:
        if i not in p and wd[i]['word'] in stopwds:
            wd.pop(i)
    return wd

stopwds = [i.strip('\n') for i in open('/media/zero/41FF48D81730BD9B/DT_RAE/Chunk_rnn/config/stopword.txt', 'r')]
wd = {}; oth={}
for i in wrd:
    if type(i) == int:
        wd[i] = wrd[i]
    else:
        oth[i] = wrd[i]
wrd = removing_stopword(wd, stopwds)
wrd.update(oth)

def syn_processing2(wrd):
    wd = {}
    for i in wrd:
        if type(i) == int:
            wd[i] = wrd[i]

    t = re.sub(r'([\)\(])', r' \1 ', wrd['tree'])
    t = re.sub(r'[ ]{2,}', r' ', t)
    t = t.split(' ')
    p = []
    wids = sorted([i for i in wd])
    ind = 0
    h_vect = []
    h_index = {}

    id=wids.pop(0)
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

_, h_vect, wp = syn_processing2(wrd)
print h_vect
print wp