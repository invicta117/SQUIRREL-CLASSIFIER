#Copyright 2019 The TensorFlow Authors.

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.



# MIT License
#
# Copyright (c) 2017 François Chollet 
# Copyright (c) 2021 Cooney                                                                                                                    # IGNORE_COPYRIGHT: cleared by OSS licensing
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Notice as per the Apache 2.0 license that the following file has been changed

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from shutil import copyfile


batch_size = 32
img_height = 160
img_width = 160

AUTOTUNE = tf.data.AUTOTUNE

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  print(parts[0])
  return parts[-1]


def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def configure_for_performance(ds):
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

model = tf.keras.models.load_model('saved_model/my_model')

final_path = os.environ["FINALPATH"]
target_path = os.environ["TARGETPATH"]

data_length = len(os.listdir(final_path))

final_ds_paths = tf.data.Dataset.list_files(final_path + "*", shuffle=False)

print(final_ds_paths)


final_ds = final_ds_paths.map(process_path, num_parallel_calls=AUTOTUNE)
final_ds = configure_for_performance(final_ds)
class_names = ["NO-SQUIRREL", "SQUIRREL"]

for final in final_ds:
    image_batch, label_batch = final
    predictions = model.predict(image_batch, batch_size=batch_size).flatten()
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    for i, label in enumerate(label_batch):
        l = label.numpy().decode('utf-8')
        print(l)
        if class_names[predictions[i]] == "SQUIRREL":
          copyfile(final_path + l, target_path + "SQUIRREL/" + l)
        else:
          copyfile(final_path + l, target_path + "NO-SQUIRREL/" + l)


print("finished?")