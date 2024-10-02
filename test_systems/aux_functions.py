import os
import re
import cv2


def save_modified_image(image, image_path):
    if os.path.isfile(image_path):
        # file already exists, no need to save it again
        return
    else:
        # Convert RGB image to BGR channel order so cv2 can save it
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
        cv2.imwrite(image_path, image)
    return


def get_modified_img_path(sample_filepath, met_rule, data_type, dataset_name):
    if dataset_name == "phoenix":
        video_path, frame_name = re.split('/images', sample_filepath)
        frame_name = "images" + frame_name
        videos_folder_path, video_name = re.split(data_type, video_path)

        modif_video_folder_path = videos_folder_path + data_type + "_" + met_rule
        modif_video_path = modif_video_folder_path + video_name
        os.makedirs(modif_video_path, exist_ok=True)

        new_image_path = modif_video_path + "/" + frame_name
        return new_image_path
    elif dataset_name == "FLIC" or dataset_name == "FLIC-full":
        video_path, frame_name = re.split('/images/', sample_filepath)
        modif_video_folder = video_path + f"/images_{met_rule}"
        os.makedirs(modif_video_folder, exist_ok=True)
        new_image_path = modif_video_folder + "/" + frame_name
        return new_image_path
    else:
        raise(ValueError, "wrong dataset name in get_modified img path")
