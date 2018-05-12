import logging
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

from paths import DATASETS_ROOT

log = logging.getLogger()

MODD_CATS = ['__background__', 'largeobjects', 'smallobjects']


class MODDLoader():
    def __init__(self, split, segmentation=False, augmented_seg=False):

        self.dataset = 'modd'

        self.root = os.path.join(DATASETS_ROOT, 'modd2/')
        self.split = split

        # assert split in ['train', 'val', 'trainval', 'test']

        cats = MODD_CATS
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]

        print('TRAINING FOR LABELS %s' % str(cats))
        self.segmentation = segmentation
        self.augmented_seg = augmented_seg

        # assert not self.segmentation or self.segmentation and self.year == '12'

        if self.augmented_seg:
            filelist = 'ImageSets/SegmentationAug/%s.txt'
        elif self.segmentation:
            filelist = 'ImageSets/Segmentation/%s.txt'
        else:
            filelist = 'ImageSets/Main/%s.txt'
        with open(os.path.join(self.root, filelist % self.split), 'r') as f:
            self.filenames = f.read().split('\n')[:-1]
        log.info("Created a loader MODD %s with %i images" % (split, len(self.filenames)))

    def load_image(self, name):
        im = Image.open('%sJPEGImages/%s.jpg' % (self.root, name)).convert('RGB')
        im = np.array(im) / 255.0
        im = im.astype(np.float32)
        return im

    def get_filenames(self):
        return self.filenames

    def read_annotations(self, name):
        bboxes = []
        cats = []

        tree = ET.parse('%sAnnotations/%s.xml' % (self.root, name))
        root = tree.getroot()
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        difficulty = []
        for obj in root.findall('object'):
            cat = self.cats_to_ids[obj.find('name').text]
            difficult = (int(obj.find('difficult').text) != 0)
            difficulty.append(difficult)
            cats.append(cat)
            bbox_tag = obj.find('bndbox')
            x = int(bbox_tag.find('xmin').text)
            y = int(bbox_tag.find('ymin').text)
            w = int(bbox_tag.find('xmax').text) - x
            h = int(bbox_tag.find('ymax').text) - y
            bboxes.append((x, y, w, h))

        gt_cats = np.array(cats)
        gt_bboxes = np.array(bboxes).reshape((len(bboxes), 4))
        difficulty = np.array(difficulty)

        seg_gt = self.read_segmentations(name, height, width)

        output = gt_bboxes, seg_gt, gt_cats, width, height, difficulty
        return output

    def read_segmentations(self, name, height, width):
        if self.segmentation:
            try:
                seg_folder = self.root + 'SegmentationClass/'
                seg_file = seg_folder + name + '.png'
                seg_map = Image.open(seg_file)
            except:
                assert self.augmented_seg
                seg_folder = self.root + 'SegmentationClassAug/'
                seg_file = seg_folder + name + '.png'
                seg_map = Image.open(seg_file)
            segmentation = np.array(seg_map, dtype=np.uint8)
        else:
            # if there is no segmentation for a particular image we fill the mask
            # with zeros to keep the same amount of tensors but don't learn from it
            segmentation = np.zeros([height, width], dtype=np.uint8)
        return segmentation
