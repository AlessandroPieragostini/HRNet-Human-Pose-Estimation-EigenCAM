
import os
import numpy as np
import shutil
import json

img_root = '.\\data\\babypose\\images\\'
anno_root = '.\\data\\babypose\\annotations\\'
li = [3, 4, 5, 6, 7, 11, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 30, 34, 36, 37, 39, 42, 43, 66, 73, 74, 76]
perc = [0.8, 0.1, 0.1] # train, val, test
images_dict = {}

tot_annotations = {'images': [],'categories': [
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
            "skeleton": [
                [
                    1,
                    2
                ],
                [
                    2,
                    3
                ],
                [
                    4,
                    5
                ],
                [
                    5,
                    6
                ],
                [
                    7,
                    8
                ],
                [
                    8,
                    9
                ],
                [
                    10,
                    11
                ],
                [
                    11,
                    12
                ]
            ]
        }
    ],'annotations':[]}

def build_annotations():
    
    for num in li:
        
        with open(anno_root + 'pz' + str(num) + '.json') as json_file:
            f = json.load(json_file)
            
            for elem in f['images']:
                new_filename = elem['path'].replace("/datasets/", "").replace("baby_pose/","").replace("/","_")
                elem["file_name"] = new_filename
                elem['path'] = img_root + images_dict[str(num)][new_filename] + "\\" + new_filename
                tot_annotations["images"].append(elem)
            tot_annotations['annotations'] += f['annotations']
    
    with open (anno_root + 'train_val_test.json', 'w') as j_file:
        json.dump(tot_annotations,j_file)

def retrieve_annotations():
    ret = {}

    for num in li:
        
        with open(anno_root + 'pz' + str(num) + '.json') as json_file:
            ret[str(num)] = []
            f = json.load(json_file)
            
            for elem in f['images']:
                ret[str(num)].append(elem['path'].replace("/datasets/", "").replace("baby_pose/",""))
    
    return ret

'''
#Splitta direttamente i pazienti in train, val e test
def split_by_pz():
    pzs = np.split(li, [int(len(li)*perc[0]), int(len(li)*(perc[0] + perc[1]))])

    for pz, mode in zip(pzs, ["train", "validation", "test"]):
        dest_dir = img_root + mode
        rmake_dir(dest_dir)
        for index in pz:
            src_dir = img_root + 'pz' + str(index)
            copy_files(os.listdir(src_dir), index, mode)
'''

#splitta le foto di ciascun paziente 80 in train, 10 val e 10 test
def split_by_perc(index, pz_files):
    pzs = np.split(pz_files, [int(len(pz_files)*perc[0]), int(len(pz_files)*(perc[0] + perc[1]))])
    for pz, mode in zip(pzs, ["train", "validation", "test"]):
        copy_files(pz,index,mode)


def copy_files(pz_files, pz_number, mode):
    dest_dir = img_root + mode

    for filename in pz_files:
        f = os.path.join(img_root, filename)
        if os.path.isfile(f):
            new_filename = filename.replace("/", "_")
            images_dict[str(pz_number)][new_filename] = mode
            shutil.copy(f, dest_dir + '\\' + new_filename)

def rmake_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

if __name__ == "__main__":
    files = retrieve_annotations()
    
    for pz in li:
        images_dict[str(pz)] = {}

    for mode in ["train", "validation", "test"]:
        rmake_dir(img_root + mode)

    for index, f in files.items():
        split_by_perc(index, f)

    build_annotations()