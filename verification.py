import numpy as np
import os
import itertools


def cal_dist(f1, f2):
    return np.sum(np.square(f1-f2))


def cal_acc(pairs, thres):
    tp = fp = tn = fn = 0
    for pair in pairs:
        idx1 = pair[0]
        idx2 = pair[1]
        s = pair[2]
        if idx1 == idx2: #positive
            if s <= thres:
                tp += 1
            else:
                fp += 1
        else:
            if s > thres:
                tn += 1
            else:
                fn += 1
    return float(tp + tn) / (tp + fp + tn + fn)


def find_best_threshold(pairs, thresholds):
    # thresholds = np.arange(0.5, 2, 0.01)
    best_thres = 0.0
    best_acc = 0.0
    for threshold in thresholds:
        acc = cal_acc(pairs, threshold)
        if acc > best_acc:
            best_acc = acc
            best_thres = threshold
        print(threshold, acc)
    return best_acc, best_thres


def cal_score(features_path, scores_path, model_list):
    for model_name in model_list:
        print(model_name)
        scores = []
        if not os.path.isdir(os.path.join(scores_path, model_name)):
            os.mkdir(os.path.join(scores_path, model_name))

        idx_feature_list = np.load(os.path.join(features_path, model_name, 'features.npy'), allow_pickle=True)
        # pair_list = list(itertools.combinations(idx_feature_list, 2))
        for i in range(len(idx_feature_list)):
            for j in range(i+1, len(idx_feature_list)):
                f1 = idx_feature_list[i][2]
                f2 = idx_feature_list[j][2]
                idx1 = idx_feature_list[i][0]
                idx2 = idx_feature_list[j][0]
                dist = cal_dist(f1, f2)
                scores.append((idx1, idx2, dist))
        print('Save scores')
        np.save(os.path.join(scores_path, model_name, 'scores.npy'), scores)


def verify(scores_path, model_list, thresholds=np.arange(0.5, 2, 0.01)):
    scores = dict()
    for model_name in model_list:
        scores[model_name] = np.load(os.path.join(scores_path, model_name, 'scores.npy'))

    file = open('veri_log.txt', 'w')
    for k, v in scores.items():
        print(k)
        file.write(k + '\n')
        file.write('total pair:' + str(len(scores[k])) + '\n')
        print('total pair', len(scores[k]))
        acc, thres = find_best_threshold(scores[k], thresholds)
        file.write('acc: ' + str(acc) + ' thres: ' + str(thres) + '\n')
        print('Best', acc, thres)
    file.close()


if __name__ == '__main__':
    features_path = './features'
    scores_path = './scores'
    model_list = ['mobilefacenet', 'resnet34', 'resnet50', 'resnet100']
    cal_score(features_path, scores_path, model_list)
    verify(scores_path, model_list)




