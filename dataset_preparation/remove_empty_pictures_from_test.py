import json
import os

import cv2

def img_is_empty(img):
    h, w = img.shape[:2]

    empty = True
    for i in range(h):
        for j in range(w):
            if any(img[i, j] != img[0, 0]):
                empty = False
                break
    return empty


if __name__ == "__main__":
    annotation_path = "fractals_2_colors/params_test_6.json"
    annotation_path_new = "fractals_2_colors/params_test_filtered_6.json"

    dataset_folder = "fractals_2_colors/"

    dct_params_new = dict()

    with open(annotation_path, "r") as json_file:
        dct_params = json.load(json_file)

    for fname in dct_params:
        fpath = os.path.join(dataset_folder, fname)

        img = cv2.imread(fpath)

        if not img_is_empty(img):
            dct_params_new[fname] = dct_params[fname]
        else:
            print(f"Image {fname} is empty")

    with open(annotation_path_new, "w") as json_file:
        json.dump(dct_params_new, json_file, indent=4)


