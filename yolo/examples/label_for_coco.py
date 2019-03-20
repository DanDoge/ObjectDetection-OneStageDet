#Huang Daoji 21/03
# To do list
#    [x] tested on jupyter nootbook
#    [ ] tested on python
#    [ ] run
#    [ ] see result

import json
import os
sys.path.insert(0, '.')
import brambox.boxes as bbb

# change to where COCO is
ROOT = "../COCO"
# where to save the .pkl files
DST = "../COCO"
# may change to 1, for there (seems to) be a slightly difference
# on bbox indexing between VOC and MSCOCO
BBOX_OFFSET = 0

# only tested on val2017, the smallest file of MSCOCO
years = [
    "2017",
]

datasets = [
    "val",
]

labels =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def get_label_for_single_file(year, dataset):
    cate = {x['id']: x['name'] for x in json.load( \
                open(f"{ROOT}/annotations/instances_{dataset}{year}.json") \
            )['categories']}

    data = json.load(open(filename))
    # parse one json file once, may need to conbine them later?
    data = json.load(open(filename))

    # hold all images
    images = {}
    for image in data["images"]:
        images[image["id"]] = {
            # add more attributes if needed
            "file_name" : image["file_name"],
            "image_size" : {
                "height": image["height"],
                "width": image["width"],
            },
            "img_id" : image["id"],
            "obj" : []
        }

    for anno in data["annotations"]:
        # attach annotations to corresponding images
        images[anno["image_id"]]["obj"].append({
            "class_label": cate[anno["category_id"]],
            "bbox": anno["bbox"]
        })

    # delete images with no annotations
    del_keys = []
    for image in images:
        if images[image]["obj"] == []:
            del_keys.append(image)
    for key in del_keys:
        del images[key]

    val_annos = {}
    for image in images:
        # change 'val' and 2017 here!
        val_annos[f'{ROOT}/val2017/{images[image]["file_name"]}'] = []
        for obj in images[image]["obj"]:
            tmp_obj = bbb.annotations.PascalVocAnnotation()
            tmp_obj.class_label = obj["class_label"]
            tmp_obj.x_top_left = int(max(obj["bbox"][0], 0))
            tmp_obj.y_top_left = int(max(obj["bbox"][1], 0))
            # maybe out of boundry! need to check?
            tmp_obj.width = int(obj["bbox"][2])
            tmp_obj.height = int(obj["bbox"][3])
            val_annos[f'{src_real_base}/val2017/{images[image]["file_name"]}'].append(tmp_obj)
    # this one contains categories not in VOC
    bbb.generate('anno_pickle', val_annos, f'{DST}/onedet_cache/MSCOCO{dataset}{year}.pkl')

    val_annos_fix_label = {}
    for image in val_annos:
        val_annos_fix_label[image] = []
        for anno in val_annos[image]:
            if anno.class_label in labels:
                val_annos_fix_label[image].append(anno)
    # this one can run like pkl generated from VOC
    bbb.generate('anno_pickle', val_annos_fix_label, f'{DST}/onedet_cache/MSCOCO2VOC{dataset}{year}.pkl')


def get_label_for_all_files():
    for year in years:
        for dataset in datasets:
            get_label_for_single_file(year, dataset)

if __name__ == '__main__':
    get_label_for_all_files()
