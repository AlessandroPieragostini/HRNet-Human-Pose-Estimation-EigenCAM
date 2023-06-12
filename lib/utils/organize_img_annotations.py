
import copy
import os
import numpy as np
import shutil
import json

img_root = '.\\data\\babypose\\images\\'
anno_root = '.\\data\\babypose\\annotations\\'
li = [3, 4, 5, 6, 7, 11, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 30, 34, 36, 37, 39, 42, 43, 66, 73, 74, 76]
images_dict = {}

perc = [0.8, 0.1, 0.1] # train, val, test
modes = ["train", "val", "test"]

template_annotations = {'images': [],'categories': [
        {
            "id": 3,
            "name": "infant",
            "supercategory": "human ",
            "color": "#616ff1",
            "metadata": {},
            "keypoint_colors": [
                "#bf5c4d",
                "#d99100",
                "#4d8068",
                "#0d2b80",
                "#9c73bf",
                "#ff1a38",
                "#bf3300",
                "#736322",
                "#33fff1",
                "#3369ff",
                "#9d13bf",
                "#733941"
            ],
            "keypoints": [
                "right_hand",
                "right_elbow",
                "right_shoulder",
                "left_shoulder ",
                "left_elbow",
                "left_hand",
                "right_foot",
                "right_knee",
                "right_hip",
                "left_hip",
                "left_knee",
                "left_foot"
            ],
            "skeleton": [ [1,2], [2,3], [4,5], [5,6], [7,8], [8,9], [10,11], [11, 12] ]
        }
    ],'annotations':[]}

def build_annotations():

    tot_annotations = {}

    for mode in modes:
        tot_annotations[mode] = copy.deepcopy(template_annotations)
    
    for num in li:

        with open(anno_root + 'pz' + str(num) + '.json') as json_file:
            f = json.load(json_file)
            for elem in f['images']:
                img_id = elem["id"]
                new_filename = str(img_id).zfill(10) + ".png"
                mode = images_dict[ int(img_id) ]
                elem["file_name"] = new_filename
                elem['path'] = img_root + mode + "\\" + new_filename
                tot_annotations[mode]["images"].append(elem)
        
            for elem in f["annotations"]:
                mode = images_dict.get( int(elem["image_id"]) )
                elem["category_id"] = 3
                if mode:
                    tot_annotations[mode]['annotations'].append(elem)
    
    for mode in modes:
        with open (f'{anno_root}person_keypoints_{mode}.json', 'w') as j_file:
            json.dump(tot_annotations[mode],j_file)

def retrieve_annotations():
    ret = {}

    for num in li:
        
        with open(anno_root + 'pz' + str(num) + '.json') as json_file:
            ret[str(num)] = []
            f = json.load(json_file)
            
            for elem in f['images']:
                filename = elem['path'].replace("/datasets/", "").replace("baby_pose/","")
                ret[str(num)].append( (filename, elem["id"]))
    
    return ret

'''
#Splitta direttamente i pazienti in train, val e test

to implement here

'''

#Splitta le foto di ciascun paziente 80 in train, 10 val e 10 test
def split_by_perc(pz_files):
    pzs = np.split(pz_files, [int(len(pz_files)*perc[0]), int(len(pz_files)*(perc[0] + perc[1]))])
    for pz, mode in zip(pzs, modes):
        copy_files(pz, mode)


def copy_files(pz_files, mode):
    dest_dir = img_root + mode

    for filename, img_id in pz_files:
        f = os.path.join(img_root, filename)
        if os.path.isfile(f):
            new_filename = str(img_id).zfill(10) + ".png"
            images_dict[ int(img_id) ] = mode
            shutil.copy(f, dest_dir + '\\' + new_filename)

def rmake_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

if __name__ == "__main__":
    files = retrieve_annotations()
    images_dict = {}

    for mode in modes:
        rmake_dir(img_root + mode)

    for item in files.items():
        split_by_perc(item[1])

    build_annotations()