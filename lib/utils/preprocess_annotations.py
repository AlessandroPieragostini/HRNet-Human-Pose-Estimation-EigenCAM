
import simplejson
import json

li = [3, 4, 5, 6, 7, 11, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 30, 34, 36, 37, 39, 42, 43, 66, 73, 74, 76]
root = '.\\data\\coco\\annotations\\'

def readAnnotations():
    
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
    
    for num in li:
        
        with open(root + 'pz' + str(num) + '.json') as json_file:
            f = json.load(json_file)
            
            for elem in f['images']:
                elem['path'] = '.\\data\\coco\\images\\' + 'pz' + str(num) + '\\' + elem['file_name']
                tot_annotations["images"].append(elem)
            tot_annotations['annotations'] += f['annotations']
    
    with open ('.\\data\\coco\\annotations\\train_val.json', 'w') as j_file:
        json.dump(tot_annotations,j_file)
        
    return 
            
if __name__ == "__main__":
    readAnnotations()