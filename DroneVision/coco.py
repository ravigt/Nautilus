import json
from collections import namedtuple
import os
from pycocotools.coco import COCO
from pathlib import Path
import numpy as np
import cv2
from ConfigLoader import load_config

AnnotationData = namedtuple('AnnotationData', ['image_file', 'img_id', 'ann_ids', 'coco'])
ImageMask = namedtuple('ImageMask', ['image_name', 'mask', 'width', 'height'])

config = load_config("config.ini")


class CocoAnnotation:

    annotation_data = dict()

    def __init__(self, root_dir):
        self.load_data(root_dir)

    def load_data(self, root_dir):
        # Specify image and annotation directories
        img_dir = Path(os.path.join(root_dir, 'images'))
        ann_dir = Path(os.path.join(root_dir, 'annotations'))

        img_files = sorted(
            [i for i in img_dir.iterdir() if not i.name.startswith(".")])
        ann_files = sorted(
            [i for i in ann_dir.iterdir() if not i.name.startswith(".")])

        coco_annotations = [COCO(ann_file) for ann_file in ann_files]

        annotations = list()
        for idx, ann in enumerate(coco_annotations):
            img_id = ann.getImgIds()[0]
            ann_ids= ann.getAnnIds()
            self.annotation_data[img_id] = AnnotationData(image_file=img_files[idx], img_id=img_id, ann_ids=ann_ids, coco=ann)

    def get_annotations(self):
        return self.annotation_data

    def get_img_by_cat(self, cat_name=[]):
        annotation_data = self.get_annotations()

        img_ids_for_category = dict(list(filter(lambda x: x[0] == x[1].coco.getImgIds(catIds=x[1].coco.getCatIds(catNms=cat_name))[0],
                                          annotation_data.items())))
        return img_ids_for_category

    def get_annotations_for_img_ids(self, img_ids: list):
        ann_list = list(map(lambda ann: ann.coco.getAnnIds(imgIds=img_ids), self.get_annotations()))
        ann_list = list(filter(lambda x: len(x) > 0, ann_list))
        return [ann for anns in ann_list for ann in anns]

    def get_images_for_img_ids(self, img_ids: list):
        for ann in self.get_annotations():
            try:
                return ann.coco.loadImgs(ids=img_ids)
            except:
                pass
        return None


def get_image_mask_for_image_id(img_id):
    # for each image get the annotation ids for road category
    annotation: COCO = img_ids_for_road[img_id].coco
    image_file = img_ids_for_road[img_id].image_file
    category_ids_for_road = annotation.getCatIds(catNms=['road'])
    if len(category_ids_for_road) == 0:
        return None

    annotation_ids_for_road = annotation.getAnnIds(catIds=category_ids_for_road)
    img = annotation.loadImgs(ids=[img_id])
    h, w = img[0]['height'], img[0]['width']
    img_name = img[0]['file_name']
    img_path = img[0]['path']
    im = np.zeros((h, w), dtype=np.uint8)

    anns = annotation.loadAnns(annotation_ids_for_road)
    areas = [i["area"] for i in anns]
    area_ids = [i for i in range(1, len(areas) + 1)][::-1]
    area_id_map = dict(zip(sorted(areas), area_ids))
    area_cat_map = {}

    for ann in anns:
        aid = area_id_map[ann["area"]]
        bMask = annotation.annToMask(ann)
        aMask = bMask * aid
        im = np.maximum(im, aMask)
        area_cat_map[aid] = ann["category_id"]

    k = np.array(list(area_cat_map.keys()))
    v = np.array(list(area_cat_map.values()))
    mapping_ar = np.zeros(k.max() + 1, dtype=np.uint8)
    mapping_ar[k] = v
    res = mapping_ar[im]
    return ImageMask(image_name=img_name, mask=res, width=w, height=h)


if __name__ == '__main__':
    coco = CocoAnnotation(config.data_path)

    # these are all the images that have roads
    img_ids_for_road = coco.get_img_by_cat(cat_name=['road'])

    for img_id in img_ids_for_road.keys():
        image_mask = get_image_mask_for_image_id(img_id)
        if image_mask is None:
            continue

        image_data_path = config.data_path+"/"+config.image_dir
        img = cv2.imread("{}/{}".format(image_data_path, image_mask.image_name))
        mask = image_mask.mask
        after_mask = cv2.bitwise_and(img, img, mask=mask)

        masked_image_data_path = config.data_path + "/" + config.masked_image_dir
        cv2.imwrite("{}/{}".format(masked_image_data_path, image_mask.image_name), after_mask)

        # if you want to see the images after application of the mask
        # uncomment the below lines
        # cv2.imshow("aa", after_mask)
        # cv2.waitKey()

        mask_data_path = config.data_path+"/"+config.mask_dir
        cv2.imwrite("{}/{}".format(mask_data_path, image_mask.image_name), mask)




