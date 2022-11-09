#based on https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
  if row_label == '__Background__':
     return 1
  elif row_label == 'person':
      return 2
  elif row_label == 'bicycle':
      return 3
  elif row_label == 'car':
      return 4
  elif row_label == 'motorcycle':
      return 5
  elif row_label == 'airplane':
      return 6
  elif row_label == 'bus':
      return 7
  elif row_label == 'train':
      return 8
  elif row_label == 'building':
      return 9
  elif row_label == 'truck':
      return 10
  elif row_label == 'boat':
      return 11
  elif row_label == 'tree':
      return 12
  elif row_label == 'trunk':
      return 13
  elif row_label == 'traffic light':
      return 14
  elif row_label == 'fire hydrant':
      return 15
  elif row_label == 'street sign':
      return 16
  elif row_label == 'stop sign':
      return 17
  elif row_label == 'parking meter':
      return 18
  elif row_label == 'bench':
      return 19
  elif row_label == 'bird':
      return 20
  elif row_label == 'cat':
      return 21
  elif row_label == 'dog':
      return 22
  elif row_label == 'horse':
      return 23
  elif row_label == 'sheep':
      return 24
  elif row_label == 'cow':
      return 25
  elif row_label == 'elephant':
      return 26
  elif row_label == 'bear':
      return 27
  elif row_label == 'zebra':
      return 28
  elif row_label == 'giraffe':
      return 29
  elif row_label == 'hat':
      return 30
  elif row_label == 'backpack':
      return 31
  elif row_label == 'umbrella':
      return 32
  elif row_label == 'shoe':
      return 33
  elif row_label == 'eye glasses':
      return 34
  elif row_label == 'handbag':
      return 35
  elif row_label == 'tie':
      return 36
  elif row_label == 'suitcase':
      return 37
  elif row_label == 'frisbee':
      return 38
  elif row_label == 'skis':
      return 39
  elif row_label == 'pole':
      return 40
  elif row_label == 'snowboard':
      return 41
  elif row_label == 'sports ball':
      return 42
  elif row_label == 'kite':
      return 43
  elif row_label == 'baseball bat':
      return 44
  elif row_label == 'baseball glove':
      return 45
  elif row_label == 'skateboard':
      return 46
  elif row_label == 'surfboard':
      return 47
  elif row_label == 'tennis racket':
      return 48
  elif row_label == 'bottle':
      return 49
  elif row_label == 'plate':
      return 50
  elif row_label == 'wine glass':
      return 51
  elif row_label == 'cup':
      return 52
  elif row_label == 'fork':
      return 53
  elif row_label == 'knife':
      return 54
  elif row_label == 'spoon':
      return 55
  elif row_label == 'bowl':
      return 56
  elif row_label == 'banana':
      return 57
  elif row_label == 'apple':
      return 58
  elif row_label == 'sandwich':
      return 59
  elif row_label == 'orange':
      return 60
  elif row_label == 'broccoli':
      return 61
  elif row_label == 'carrot':
      return 62
  elif row_label == 'hot dog':
      return 63
  elif row_label == 'pizza':
      return 64
  elif row_label == 'donut':
      return 65
  elif row_label == 'cake':
      return 66
  elif row_label == 'chair':
      return 67
  elif row_label == 'couch':
      return 68
  elif row_label == 'potted plant':
      return 69
  elif row_label == 'bed':
      return 70
  elif row_label == 'mirror':
      return 71
  elif row_label == 'dining table':
      return 72
  elif row_label == 'window':
      return 73
  elif row_label == 'desk':
      return 74
  elif row_label == 'toilet':
      return 75
  elif row_label == 'door':
      return 76
  elif row_label == 'tv':
      return 77
  elif row_label == 'laptop':
      return 78
  elif row_label == 'mouse':
      return 79
  elif row_label == 'remote':
      return 80
  elif row_label == 'keyboard':
      return 81
  elif row_label == 'cell phone':
      return 82
  elif row_label == 'microwave':
      return 83
  elif row_label == 'oven':
      return 84
  elif row_label == 'toaster':
      return 85
  elif row_label == 'sink':
      return 86
  elif row_label == 'refrigerator':
      return 87
  elif row_label == 'blender':
      return 88
  elif row_label == 'book':
      return 89
  elif row_label == 'clock':
      return 90
  elif row_label == 'vase':
      return 91
  elif row_label == 'scissors':
      return 92
  elif row_label == 'teddy bear':
      return 93
  elif row_label == 'hair drier':
      return 94
  elif row_label == 'toothbrush':
      return 95
  elif row_label == 'hair brush':
      return 96
  else:
      return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()