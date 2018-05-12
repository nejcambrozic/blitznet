from glob import glob
from os import path as osp

import numpy as np
import tensorflow as tf
from PIL import Image

from config import args, train_dir
from config import config as net_config
from detector import Detector
from modd_loader import MODD_CATS
from paths import DEMO_DIR
from resnet import ResNet
from voc_loader import VOC_CATS

slim = tf.contrib.slim


# VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#            'tvmonitor']

# MODD_CATS = ['__background__', 'largeobjects', 'smallobjects']

class Loader():
    def __init__(self, folder=DEMO_DIR, data_format='.jpg'):
        cats = MODD_CATS  # VOC_CATS #
        self.folder = folder
        self.data_format = data_format
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]
        self.filenames = self.load_filenames()

    def load_filenames(self):
        files = glob(osp.join(self.folder, '*{}'.format(self.data_format)))
        filenames = [n.split('/')[-1][:-len(self.data_format)] for n in files]
        return filenames

    def load_image(self, name):
        im = Image.open(osp.join(self.folder, name + self.data_format)).convert('RGB')
        im = np.array(im) / 255.0
        im = im.astype(np.float32)
        return im

    def get_filenames(self):
        return self.filenames


def main(argv=None):  # pylint: disable=unused-argument
    assert args.detect or args.segment, "Either detect or segment should be True"
    assert args.ckpt > 0, "Specify the number of checkpoint"
    net = ResNet(config=net_config, depth=50, training=False)
    loader = Loader()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        detector = Detector(sess, net, loader, net_config, no_gt=args.no_seg_gt,
                            folder=osp.join(loader.folder, 'output'))
        print('Restore ckpt arguments ', args.ckpt)
        detector.restore_from_ckpt(args.ckpt)
        for name in loader.get_filenames():
            image = loader.load_image(name)
            h, w = image.shape[:2]
            print('Processing {}'.format(name + loader.data_format))
            detector.feed_forward(img=image, name=name, w=w, h=h, draw=True,
                                  seg_gt=None, gt_bboxes=None, gt_cats=None)
    print('Done')


if __name__ == '__main__':
    tf.app.run()
