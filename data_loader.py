from glob import glob
import numpy as np
from scipy.misc import imread, imresize, imsave
from rng import *


def load_random_samples(dataset_name, batch_size, fine_size):
    filenames = np_rng.choice(glob('./datasets/%s/val/*.jpg' % (dataset_name)), batch_size)
    sample = [load_data_test(file_name, fine_size) for file_name in filenames]
    return sample


def load_data(file_name, load_size, fine_size, flip):
    img = imread(file_name)
    w = int(img.shape[1])
    w2 = int(w / 2)
    imgB, imgA = img[:, 0:w2], img[:, w2:]

    imgA = imresize(imgA, [load_size, load_size])
    imgB = imresize(imgB, [load_size, load_size])

    h1 = int(np.ceil(np_rng.uniform(1e-2, load_size - fine_size)))
    w1 = int(np.ceil(np_rng.uniform(1e-2, load_size - fine_size)))
    imgA = imgA[h1:h1 + fine_size, w1:w1 + fine_size]
    imgB = imgB[h1:h1 + fine_size, w1:w1 + fine_size]

    if flip and np_rng.rand() > 0.5:
        imgA = np.fliplr(imgA)
        imgB = np.fliplr(imgB)

    imgA = imgA / 127.5 - 1.
    imgB = imgB / 127.5 - 1.
    imgAB = np.concatenate((imgA, imgB), axis=2)

    return imgAB


def load_data_test(file_name, fine_size):
    img = imread(file_name)
    w = int(img.shape[1])
    w2 = int(w / 2)
    imgB, imgA = img[:, 0:w2], img[:, w2:]

    imgA = imresize(imgA, [fine_size, fine_size])
    imgB = imresize(imgB, [fine_size, fine_size])

    imgA = imgA / 127.5 - 1.
    imgB = imgB / 127.5 - 1.
    imgAB = np.concatenate((imgA, imgB), axis=2)

    return imgAB


def data_provider(dataset_name, batch_size, load_size, fine_size, flip, shuffle):
    file_names = glob('./datasets/%s/train/*.jpg' % (dataset_name))
    # batch_idxs = min(len(file_names), train_size) // batch_size
    batch_idxs = len(file_names) // batch_size
    if shuffle:
        np_rng.shuffle(file_names)

    for idx in range(0, batch_idxs):
        batch_file_names = file_names[idx * batch_size: (idx + 1) * batch_size]
        batch = [load_data(file_name, load_size, fine_size, flip) for file_name in batch_file_names]
        batch = np.asarray(batch, dtype=np.float32)
        yield batch


def data_provider_test(dataset_name, batch_size, fine_size):
    file_names = glob('./datasets/%s/train/*.jpg' % (dataset_name))
    # batch_idxs = min(len(file_names), train_size) // batch_size
    batch_idxs = len(file_names) // batch_size

    for idx in range(0, batch_idxs):
        batch_file_names = file_names[idx * batch_size: (idx + 1) * batch_size]
        batch = [load_data_test(file_name, fine_size) for file_name in batch_file_names]
        batch = np.asarray(batch, dtype=np.float32)
        yield batch


def save_images(imgs, samples, sample_dir, iter_counter, input_nc=3):
    for i, (img, sample) in enumerate(zip(imgs, samples)):
        imgA, imgB = img[:, :, :input_nc], img[:, :, input_nc:]
        imgA = (imgA + 1.) * 127.5
        imgB = (imgB + 1.) * 127.5
        sample = (sample + 1.) * 127.5
        image = np.concatenate((imgA, imgB, sample), axis=1)
        path = sample_dir + '/sample_%d_%d.jpg' % (iter_counter, i)
        imsave(path, image)
