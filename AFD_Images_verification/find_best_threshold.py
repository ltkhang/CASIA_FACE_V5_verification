import sys
import os
import numpy as np
import time


def cal_tp_tn(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))

    # acc = float(tp + tn) / len(dist)
    return tp, tn


if __name__ == '__main__':
    t1= time.time()
    model_name = sys.argv[1]
    print(model_name)
    scores_path = './scores'
    acc = dict()
    thresholds = np.arange(0.5, 2.0, 0.01)
    for t in thresholds:
        if t not in acc:
            acc[t] = dict()
            acc[t]['tp'] = 0
            acc[t]['tn'] = 0
    f = open(os.path.join(scores_path, model_name, 'scores.txt'), 'r')
    i = 0
    MAX = 10000000
    dist = []
    actual_issame = []
    total = 0
    t2 = time.time()
    while True:
        line = f.readline()
        if not line:
            total += i
            for t in thresholds:
                tp, tn = cal_tp_tn(t, dist, actual_issame)
                acc[t]['tp'] += tp
                acc[t]['tn'] += tn
            del dist
            del actual_issame
            dist = []
            actual_issame = []
            break
        i += 1
        d = line.strip()
        is_same, str_score = d.split(',')
        dist.append(float(str_score))
        if is_same == '1':
            actual_issame.append(True)
        else:
            actual_issame.append(False)
        if i == MAX:
            total += i
            i = 0

            for t in thresholds:
                tp, tn = cal_tp_tn(t, dist, actual_issame)
                acc[t]['tp'] += tp
                acc[t]['tn'] += tn
            print('cal tp tn', time.time() - t2)
            del dist
            del actual_issame
            dist = []
            actual_issame = []
        print(i)

    f.close()
    best_thres = 0.0
    best_acc = 0.0
    print('total', total)
    f = open('log_' + model_name + '.txt', 'w')
    for thres, _ in acc.items():
        tp = acc[thres]['tp']
        tn = acc[thres]['tn']
        accuracy = float(tp + tn) / total
        if accuracy > best_acc:
            best_thres = thres
            best_acc = accuracy
        f.write('{},{}\n'.format(thres, accuracy))
    f.write('Best: {},{}'.format(best_thres, best_acc))
    f.close()
    print('time:', time.time() - t1)