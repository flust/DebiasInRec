import os, sys

ds = sys.argv[1]
mode = sys.argv[2]

def get_record(root, mode):
    f_path = os.path.join(root, "%s.record"%mode)
    logloss = list()
    auc = list()
    with open(f_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith(('ext', 'bi' ,'ffm', '0.')):
                logloss.append(line.strip().split(' ')[-2])
                auc.append(line.strip().split(' ')[-1])
    return logloss, auc

roots = ['cbias/derive.det.ffm.wops'] + ['normal/'+i for i in 'derive.det.extffm.wops derive.det.biffm.wops derive.random.ffm.wops derive.det.ffm.wops der.comb.0.01.ffm.wops der.comb.0.1.ffm.wops'.split()]
roots = ['run_dl_exp/%s/'+i for i in roots] + ['run_fm_exp/%s/normal/'+i for i in 'der.comb.0.01.imp.pw der.comb.0.1.imp.pw derive.det.ps der.comb.0.01.ps der.comb.0.1.ps derive.random.ps der.comb.0.01.imp.pspw der.comb.0.1.imp.pspw'.split()] + ['run_fm_exp/%s/RD/derive.det.select.ps', 'run_fm_exp/%s/DR/derive.det.select.ps']
tags = 'FFM-Ideal FFM-Greedy-A FFM-Greedy-P FFM-Random FFM-Greedy FFM-EE(0.01) FFM-EE(0.1) FFM-CF(0.01) FFM-CF(0.01) TFM-Greedy TFM-EE(0.01) TFM-EE(0.1) TFM-Random TFM-CF(0.01) TFM-CF(0.1) TFM-RG TFM-GR'.split()

print('method,ll-pos,auc-pos,ll-alpha,auc-alpha')
for i, root in enumerate(roots):
    l, a = get_record(root%ds, mode)
    if 'TFM' not in tags[i]:
        print(','.join([tags[i], l[-1], a[-1], l[1], a[1]]))
    else:
        if 'Greedy' in tags[i]:
            print('='*20)
        print(','.join([tags[i], l[-1], a[-1], '--', '--']))

