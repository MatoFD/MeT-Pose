import numpy as np


def pck3d(landmarks_a, landmarks_b, normalization="pose"):
    """ count number of key points that the visibility matches and absolute 2D Euclidean error between the original and
     modified keypoint is lower than the 20% of the 3D torso diameter.

    both landmarks arrays should be an array of all the keypoints localized for a single image. Before calling, it should
    be checked that both arrays are not None.

    A score of 100 means that all keypoints are "correct", a score of 65 means that 65% of keypoints match well enough
    to be considered "correct", and so on

    see https://stasiuk.medium.com/pose-estimation-metrics-844c07ba0a78,
    https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf and
    https://github.com/cbsudux/Human-Pose-Estimation-101 for more information

    :param landmarks_a: should be the more "trustworthy" of the two landmarks, for normalizing
    :param landmarks_b: the landmarks we want to compare against the original ones
    :param normalization: Chooses if we normalize by the diameter of the torso, the head bone link, or the bounding box
    diagonal
    :return:
    """
    assert len(landmarks_a) == len(landmarks_b)

    def distance(vector):
        """get Euclidean length of input vector, could be changed to L1 norm or others
        """
        return np.linalg.norm(vector, ord=2)

    # the mediapipe model cards for hands and face specify that they evaluate the model over 2D without the z dimension,
    # because the ground truths of human annotations only have 2D, so we ignore the z dimension for now.

    # Only keep the landmarks that have a close predicted visibility (10% error margin), this was an arbitrary
    # number, but the pose model card specifies to look at the visibility match without specifying a number.
    # each coordinate is of the form: (x, y, visibility) as translated in test_mediapipe_holistic.mediapipe_to_dict
    if normalization == "torso":
        distances = np.array([
            distance((a[0] - b[0], a[1] - b[1])) for a, b in zip(landmarks_a, landmarks_b)
            if abs(a[2] - b[2]) <= 0.1  # if visibility doesn't match, don't count this landmark as correctly detected
        ])
    elif normalization == "flic_gt_torso":
        # When comparing with flic ground truth, we don't have any visibility ground truth
        distances = np.array([
            distance((a[0] - b[0], a[1] - b[1])) for a, b in zip(landmarks_a, landmarks_b)
        ])
    else:
        raise "please specify the way to decide if the error of the landmarks is acceptable or not"

    # for each norm distance, we normalize the obtained value between sample by dividing it with the original
    # measured distance diameter of the torso (distance between shoulders) * 0.2
    # We only normalize based on landmarks_a distance because landmarks_a is the more "trustworthy one", either ground
    # truth or the original before metamorphic rules image. So if the non trustworthy is very bad, we can still trust
    # the normalization somewhat.
    if normalization == "pose":
        # https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
        shoulder_left = landmarks_a[12]
        shoulder_right = landmarks_a[11]
        normalization_value = distance(
            (shoulder_left[0] - shoulder_right[0], shoulder_left[1] - shoulder_right[1])
        )
        normalization_value *= 0.2
    elif normalization == "flic_gt_torso":
        # if we are comparing mediapipe with flic ground truth annotations, flic has only 12 keypoints, so we only
        # grab those 12 out of holistic pose landmarks, so we need to grab different indexes than when comparing
        # two different mediapipe outputs. But the computation is the same as the "torso" option
        # The order comes from test_mediapipe_holistic.diff_holistic_pose_and_flic_ground_truth foo before calling pck3d
        shoulder_left = landmarks_a[0]
        shoulder_right = landmarks_a[3]
        normalization_value = distance(
            (shoulder_left[0] - shoulder_right[0], shoulder_left[1] - shoulder_right[1])
        )
        normalization_value *= 0.2
    else:
        raise "please specify the way to normalize the error of the landmarks"

    # count number of keypoint distances that are smaller that the normalized threshold as "correct"
    num_detected = (distances < normalization_value).sum()

    percentage = (num_detected * 100) / len(landmarks_a)
    return percentage


def normalized_mean_absolute_error(landmarks_a, landmarks_b, normalization):
    diffs = get_2d_landmark_diffs(landmarks_a, landmarks_b, normalization)
    return np.sum(diffs) / len(landmarks_a)


def get_2d_landmark_diffs(landmarks_a, landmarks_b, normalization):
    """This NMAE function was made mainly to test mediapipe holistic implementation. As such, many decisions were made
    to replicate the evaluation method described in their model cards, including using knowledge of the order of the
    landmarks returned by the model, and probably will not work as is for other models.

    both landmarks arrays should be an array of all the keypoints localized for a single image. Before calling, it should
    be checked that both arrays are not None.

    An error of 1 means that the keypoints are as far apart as the distance between the eyes in a face, the
    distance between the wrist and middle finger in a hand, or the shoulders in the whole body pose.

    :param landmarks_a: should be the more "trustworthy" of the two landmarks, for normalizing
    :param landmarks_b: the landmarks we want to compare against the original ones
    :param normalization:
    :return:
    """
    assert len(landmarks_a) == len(landmarks_b)

    # We measure Euclidean distance between the two predicted landmarks, could be changed to L1 norm or others
    def distance(vector):
        return np.linalg.norm(vector, ord=2)

    # the mediapipe model cards for hands and face specify that they evaluate the model over 2D without the z dimension,
    # because the ground truths of human annotations only have 2D, so we ignore the z dimension.
    distances = np.array([
        distance((a[0] - b[0], a[1] - b[1])) for a, b in zip(landmarks_a, landmarks_b)
    ])

    # for each norm distance, we normalize the obtained value between sample by dividing it with the original
    # measured distance between the wrist and the first joint (MCP) of the middle finger or the distance between the
    # eye centers following the mediapipe model cards.
    # We only normalize based on landmarks_a distance because landmarks_a is the more "trustworthy one", either ground
    # truth or the original before metamorphic rules image. So if the non trustworthy is very bad, we can still trust
    # the normalization somewhat.
    if normalization == "hands":
        # 0 is wrist, 9 is MCP of middle finger.
        # https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md#hand-landmark-model
        wrist = landmarks_a[0]
        middle_finger = landmarks_a[9]
        normalization_value = distance((wrist[0] - middle_finger[0], wrist[1] - middle_finger[1]))
    elif normalization == "face":
        # face landmarks gotten from https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        # https://github.com/google/mediapipe/issues/1615  and  https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
        left_eye = ((landmarks_a[362][0] + landmarks_a[263][0]) / 2,
                    (landmarks_a[362][1] + landmarks_a[263][1]) / 2)  # avg 362 y 263 corners of the eye
        right_eye = ((landmarks_a[33][0] + landmarks_a[133][0]) / 2,
                     (landmarks_a[33][1] + landmarks_a[133][1]) / 2)  # avg 33 y 133 corners of the eye
        normalization_value = distance((left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]))
    elif normalization == "pose":
        # https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
        shoulder_left = landmarks_a[12]
        shoulder_right = landmarks_a[11]
        normalization_value = distance(
            (shoulder_left[0] - shoulder_right[0], shoulder_left[1] - shoulder_right[1])
        )
    elif normalization == "flic_gt_torso":
        # if we are comparing mediapipe with flic ground truth annotations, flic has only 12 keypoints, so we only
        # grab those 12 out of holistic pose landmarks, so we need to grab different indexes than when comparing
        # two different mediapipe outputs. But the computation is the same as the "torso" option
        # The order comes from test_mediapipe_holistic.diff_holistic_pose_and_flic_ground_truth foo before calling pck3d
        shoulder_left = landmarks_a[0]
        shoulder_right = landmarks_a[3]
        normalization_value = distance(
            (shoulder_left[0] - shoulder_right[0], shoulder_left[1] - shoulder_right[1])
        )
    else:
        raise ValueError(f"please specify the way to normalize the error of the landmarks, received {normalization}")

    # normalization_value is an integer, the normalization distance in this specific image, it is not an array
    distances /= normalization_value

    return distances
