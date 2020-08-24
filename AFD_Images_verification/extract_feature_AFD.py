import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import extract_feature

if __name__ == '__main__':
    dataset_dir = './aligned_faces'
    models_path = '../models'
    features_path = './features'
    extract_feature.run(dataset_dir, models_path, features_path, 'jpg')