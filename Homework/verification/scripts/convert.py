#!/usr/bin/python

# Copyright (c) 2014, Yaroslav Ganin (yaroslav.ganin@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import os
import shutil
import argparse
import struct
from array import array as pyarray
import lmdb

import numpy as np
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/Users/acid/Yandex.Disk/Documents/shad/year_3/term2/Deep_Learning/caffe'  # Path to Caffe root folder
import sys
sys.path.insert(0, caffe_root + 'python')

from caffe.proto.caffe_pb2 import Datum

def read_mnist(dataset='train', path='.', data_range=None):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "Dataset must be 'test' or 'train', not '%s'" % dataset

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    if data_range is not None:
        num_elems = data_range[1] - data_range[0]
        flbl.seek(data_range[0], 1)
        labels = np.fromfile(flbl, dtype='>u1', count=num_elems).reshape((num_elems, 1))
    else:
        labels = np.fromfile(flbl, dtype='>u1').reshape((size, 1))
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    if data_range is not None:
        num_elems = data_range[1] - data_range[0]
        fimg.seek(data_range[0] * rows * cols, 1)
        images = np.fromfile(fimg, dtype='>u1', count=num_elems * rows * cols).reshape((num_elems, rows, cols))
    else:
        images = np.fromfile(fimg, dtype='>u1').reshape((size, rows, cols))
    fimg.close()

    images = np.require(images, requirements='C')
    labels = np.require(labels, requirements='C')

    return images, labels

def convert_dataset(source_path, mode, dilate, target_path):
    images, labels = read_mnist(mode, source_path)

    if dilate == 1:
        suffix = 'dilated_'
    else:
        suffix = ''

    db_path = os.path.join(target_path, '{0}mnist_{1}_lmdb'.format(suffix, mode))

    try:
        shutil.rmtree(db_path)
    except:
        pass
    os.makedirs(db_path, mode=0744)

    num_images = images.shape[0]

    datum = Datum();
    datum.channels = 1
    datum.height = images.shape[1]
    datum.width = images.shape[2]

    mdb_env = lmdb.Environment(db_path, map_size=1099511627776, mode=0664)
    mdb_txn = mdb_env.begin(write=True)
    mdb_dbi = mdb_env.open_db(txn=mdb_txn)

    for i in xrange(num_images):
        img = images[i, :, :]
        if dilate == 1:
            img = cv2.dilate(img, np.ones((3, 3)))

        datum.data = img.tostring()
        datum.label = np.int(labels.ravel()[i])

        value = datum.SerializeToString()
        key = '{:08d}'.format(i)

        mdb_txn.put(key, value, db=mdb_dbi)

        if i % 1000 == 0:
            mdb_txn.commit()
            mdb_txn = mdb_env.begin(write=True)

    if num_images % 1000 != 0:
        mdb_txn.commit()

    mdb_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts MNIST into lmdb/caffe format optionally dilating images')
    parser.add_argument('-m', '--mode', choices=['train', 'test'], required=True)
    parser.add_argument('-s', '--source-path', dest='source_path', required=True)
    parser.add_argument('-t', '--target-path', dest='target_path', required=True)
    parser.add_argument('-d', '--dilate', action='store_true', default=False)

    args = parser.parse_args()

    convert_dataset(args.source_path, args.mode, args.dilate, args.target_path)
