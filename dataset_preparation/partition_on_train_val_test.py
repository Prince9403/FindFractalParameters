import json


if __name__ == "__main__":
    path_to_annotations = "fractals_2_colors/params.json"

    # we will try to use several train ratios and compare results
    train_ratios = [0.4, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9]

    validation_ratio = 0.2

    with open(path_to_annotations, "r") as fd:
        dct_annotations = json.load(fd)

    for i, train_ratio in enumerate(train_ratios):

        dct_annotations_train = dict()
        dct_annotations_val = dict()
        dct_annotations_test = dict()

        partition_num_test = int(train_ratio * len(dct_annotations))
        partition_num_validation = int(partition_num_test * (1.0 - validation_ratio))

        for image_num in dct_annotations.keys():
            image_full_name = f"img_{image_num}.jpg"
            if int(image_num) < partition_num_validation:
                dct_annotations_train[image_full_name] = dct_annotations[image_num]
            elif int(image_num) < partition_num_test:
                dct_annotations_val[image_full_name] = dct_annotations[image_num]
            else:
                dct_annotations_test[image_full_name] = dct_annotations[image_num]

        with open(f"fractals_2_colors/params_train_{i + 1}.json", "w") as fd:
            json.dump(dct_annotations_train, fd, indent=4)

        with open(f"fractals_2_colors/params_val_{i + 1}.json", "w") as fd:
            json.dump(dct_annotations_val, fd, indent=4)

        with open(f"fractals_2_colors/params_test_{i + 1}.json", "w") as fd:
            json.dump(dct_annotations_test, fd, indent=4)

