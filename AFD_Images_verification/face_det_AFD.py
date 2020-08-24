import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import face_det


if __name__ == '__main__':
    dataset_dir = './AFD_Images'
    faces_pose_path = './faces_pose'
    dest_dir = './aligned_faces'
    face_det.run(dataset_dir, faces_pose_path, dest_dir, '../mtcnn-model', 'jpg')
