import numpy as np


def pivot(d, bn, s):
    l = d.tolist()[0][:-2]
    jnum = l.index(max(l)) # Номер перевода
    m = []
    for i in range(bn):
        if d[i, jnum] == 0:
            m.append(0.)
        else:
            m.append(d[i, -1]/d[i, jnum])
    inum = m.index(min([x for x in m[1:] if x!=0]))  # Перенести нижний индекс
    s[inum-1] = jnum
    r = d[inum, jnum]
    d[inum] /= r
    for i in [x for x in range(bn) if x !=inum]:
        r = d[i, jnum]
        d[i] -= r * d[inum]        


def solve(d, bn, cn):
    s = list(range(cn-bn,cn-1))
    flag = True
    while flag:
        if max(d.tolist()[0]) <= 0: # Пока все коэффициенты не будут меньше или равны 0
            flag = False
        else:
            pivot(d, bn, s)            

    return s


def printSol(d, cn, s):
    for i in range(cn - 1):
        if i in s:
            print("x"+str(i)+"=%.2f" % d[s.index(i)+1, -1])
        else:
            print("x"+str(i)+"=0.00")
    print("objective is %.2f"%(-d[0, -1]))
