import cv2
import numpy as np
import mxnet as mx
import sklearn
from face_det import get_idx_list
import os
import itertools
import time
from sklearn.preprocessing import normalize


def get_model(model_path, ctx, image_size=112):
    layer = 'fc1'
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path + '/model', 0)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size, image_size))])
    model.set_params(arg_params, aux_params)
    return model


def get_feature(model, nimg):
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = normalize(embedding).flatten()
    return embedding


def run(dataset_dir, models_path, features_path, ext='bmp'):
    ctx = mx.gpu(0)
    idx_list, image_idx_list = get_idx_list(dataset_dir, ext)
    print('Total id', len(idx_list))
    print('Total images: ', len(image_idx_list))
    # load image
    image_dict = dict()
    for image_idx in image_idx_list:
        idx = image_idx[0]
        image_name = image_idx[1]
        filepath = os.path.join(dataset_dir, idx, image_name)
        key = '{}_{}'.format(idx, image_name)
        image_dict[key] = cv2.imread(filepath)

    model_list = ['mobilefacenet', 'resnet34', 'resnet50', 'resnet100']
    result = dict()
    file = open('log.txt', 'w')
    for model_name in model_list:
        total_time = 0
        if not os.path.isdir(os.path.join(features_path, model_name)):
            os.mkdir(os.path.join(features_path, model_name))

        print(model_name)
        file.write(model_name + '\n')
        result[model_name] = []
        model = get_model(os.path.join(models_path, model_name), ctx)
        for image_idx in image_idx_list:
            idx = image_idx[0]
            image_name = image_idx[1]
            key = '{}_{}'.format(idx, image_name)
            img = image_dict[key]
            t1 = time.time()
            f = get_feature(model, img)
            total_time += time.time() - t1
            result[model_name].append((idx, image_name, f))
        print('Avg time', 1000 * total_time / len(image_idx_list), 'ms')
        print('Avg face per second', 1 / (total_time / len(image_idx_list)))
        file.write('Avg time: ' + str(1000 * total_time / len(image_idx_list)) + ' ms\n')
        file.write('Avg face per second: ' + str(1 / (total_time / len(image_idx_list))) + '\n')
        del model
    file.close()

    print('Save feature')
    for k, v in result.items():
        np.save(os.path.join(features_path, k, 'features.npy'), np.asarray(result[k], dtype=object))


if __name__ == '__main__':
    dataset_dir = './aligned_faces'
    models_path = './models'
    features_path = './features'
    run(dataset_dir, models_path, features_path)

