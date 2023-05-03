import numpy as np
def otsu_threshold(data, num_grids=100):
    '''
    :param data: 1d-array
    :return: threshold that maximizes inter-class variance (p1*p2*(m1-m2)**2)
    '''
    m, M = min(data), max(data)
    otsu_tau, otsu_var = -1, -1
    for th in np.linspace(start=m+1e-12, stop=M, num=num_grids, endpoint=False):
        c1, c2 = data[data<th], data[data>th]
        p1, p2 = len(c1)/len(data), len(c2)/len(data)
        m1, m2 = c1.mean(), c2.mean()
        icvar = (p1*p2*(m1-m2)**2)
        if icvar > otsu_var:
            otsu_var = icvar
            otsu_tau = th
    return otsu_tau