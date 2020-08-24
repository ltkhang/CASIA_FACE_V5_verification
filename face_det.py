import os
from glob import glob
import cv2
import mxnet
import sklearn
import face_preprocess
from mtcnn_detector import MtcnnDetector
import numpy as np
from head_pose import detect_head_pose
import json


def get_idx_list(dataset_dir, ext='bmp'):
    idx_list = [o for o in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, o))]
    image_idx_list = []
    for idx in sorted(idx_list):
        for image_path in glob(os.path.join(dataset_dir, idx, '*.' + ext)):
            image_idx_list.append((idx, os.path.basename(image_path)))
    return idx_list, image_idx_list


def get_detector(ctx, mtcnn_path, ):
    return MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                         threshold=[0.6, 0.7, 0.8])


def get_input(detector, face_img):
    ret = detector.detect_face(face_img)
    if ret is None:
      return None, None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None, None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    # aligned = np.transpose(nimg, (2,0,1))
    #pose
    pitch, yaw, roll = detect_head_pose(face_img.shape, bbox, points)
    return nimg, roll


def run(dataset_dir, faces_pose_path, dest_dir, mtcnn_model_dir, ext='bmp'):
    idx_list, image_idx_list = get_idx_list(dataset_dir, ext)
    print('Total id', len(idx_list))
    print('Total images: ', len(image_idx_list))
    np.save('image_idx_list.npy', np.asarray(image_idx_list))
    ctx = mxnet.gpu(0)
    detector = get_detector(ctx, mtcnn_model_dir)

    roll_res = dict()
    for idx in idx_list:
        if not os.path.isdir(os.path.join(dest_dir, idx)):
            os.mkdir(os.path.join(dest_dir, idx))
        roll_res[idx] = dict()

    for image_idx in image_idx_list:
        idx = image_idx[0]
        img = cv2.imread(os.path.join(dataset_dir, image_idx[0], image_idx[1]))
        aligned_face, roll = get_input(detector, img)
        if aligned_face is not None:
            cv2.imwrite(os.path.join(dest_dir, image_idx[0], image_idx[1]), aligned_face)
            if 'roll' not in roll_res[idx] or abs(roll) < abs(roll_res[idx]['roll']):
                roll_res[idx]['roll'] = roll
                roll_res[idx]['img_path'] = image_idx[1]
        else:
            print(image_idx)

    json.dump(roll_res, open(os.path.join(faces_pose_path, 'roll.json'), 'w'))

    idx_aligned_list, image_idx_aligned_list = get_idx_list(dest_dir, ext)
    print('Total id', len(idx_aligned_list))
    print('Total aligned images: ', len(image_idx_aligned_list))


if __name__ == '__main__':
    dataset_dir = './CASIA_FACE_V5'
    faces_pose_path = './faces_pose'
    dest_dir = './aligned_faces'
    run(dataset_dir, faces_pose_path, dest_dir, './mtcnn-model')




