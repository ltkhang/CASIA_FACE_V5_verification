import numpy as np
import os
import itertools
import json
from verification import cal_dist


if __name__ == '__main__':
    faces_pose_path = './faces_pose'
    threshold_list = dict()
    model_list = ['mobilefacenet', 'resnet34', 'resnet50', 'resnet100']
    threshold_list['mobilefacenet'] = 0.82
    threshold_list['resnet34'] = 0.86
    threshold_list['resnet50'] = 0.86
    threshold_list['resnet100'] = 0.88

    roll_pose = json.load(open(os.path.join(faces_pose_path, 'roll.json'), 'r'))
    aligned_faces_path = './aligned_faces'
    features_path = './features'

    for model_name in model_list:
        print(model_name)
        idx_feature_list = np.load(os.path.join(features_path, model_name, 'features.npy'), allow_pickle=True)
        pair_list = list(itertools.combinations(idx_feature_list, 2))
        t = 0
        f = 0
        threshold = threshold_list[model_name]
        for pair in pair_list:
            f1 = pair[0][2]
            f2 = pair[1][2]
            image_idx1 = pair[0][1]
            image_idx2 = pair[1][1]
            idx1 = pair[0][0]
            idx2 = pair[1][0]
            dist = cal_dist(f1, f2)
            if roll_pose[idx1]['img_path'] == image_idx1 or roll_pose[idx2]['img_path'] == image_idx2: #frontal_face
                # pair_score.append((idx1, idx2, dist))
                if idx1 == idx2: #same person
                    if dist <= threshold:
                        t += 1
                    else:
                        f += 1
                else:
                    if dist > threshold:
                        t += 1
                    else:
                        f += 1
        acc = float(t) / float(t + f)
        print(acc)

