import os
from glob import glob
import cv2
import mxnet
import sklearn
import face_preprocess
from mtcnn_detector import MtcnnDetector
import numpy as np


def get_idx_list(dataset_dir):
    idx_list = [o for o in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, o))]
    image_idx_list = []
    for idx in sorted(idx_list):
        for image_path in glob(os.path.join(dataset_dir, idx, '*.bmp')):
            image_idx_list.append((idx, os.path.basename(image_path)))
    return idx_list, image_idx_list


def get_detector(ctx, mtcnn_path, ):
    return MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                         threshold=[0.6, 0.7, 0.8])


def get_input(detector, face_img):
    ret = detector.detect_face(face_img)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    # aligned = np.transpose(nimg, (2,0,1))
    return nimg


if __name__ == '__main__':
    dataset_dir = './CASIA_FACE_V5'
    idx_list, image_idx_list = get_idx_list(dataset_dir)
    print('Total id', len(idx_list))
    print('Total images: ', len(image_idx_list))
    np.save('image_idx_list.npy', np.asarray(image_idx_list))
    ctx = mxnet.gpu(0)
    detector = get_detector(ctx, './mtcnn-model')
    dest_dir = './aligned_faces'
    for idx in idx_list:
        if not os.path.isdir(os.path.join(dest_dir, idx)):
            os.mkdir(os.path.join(dest_dir, idx))
    for image_idx in image_idx_list:
        img = cv2.imread(os.path.join(dataset_dir, image_idx[0], image_idx[1]))
        aligned_face = get_input(detector, img)
        if aligned_face is not None:
            cv2.imwrite(os.path.join(dest_dir, image_idx[0], image_idx[1]), aligned_face)
        else:
            print(image_idx)

    idx_aligned_list, image_idx_aligned_list = get_idx_list(dest_dir)
    print('Total id', len(idx_aligned_list))
    print('Total aligned images: ', len(image_idx_aligned_list))




