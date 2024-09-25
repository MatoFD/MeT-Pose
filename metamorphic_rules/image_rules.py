import argparse
import math
import os

import numpy as np
import cv2

import mediapipe as mp

from test_systems.aux_functions import save_modified_image, get_modified_img_path
from .metamorphic_rules import MetRule



class ImgBlurRule(MetRule):
    """Applies a bilateral filter to blur the image, intended to test how sensitive the system is to textures,
        while maintaining the information of the shapes of the image.
    """
    def __init__(self, **kwargs):
        """Blur Metamorphic Rule.

        Applies a blur filter to the image, intended to test how sensitive the system is to textures, while maintaining
        the information of the shapes of the image. We use bilateral filtering, following
        https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

        :param kwargs:
            :param blur_strength:
        """
        super().__init__(**kwargs)
        self.blur_strength = kwargs['blur_strength']
        self.filter_size = kwargs['filter_size']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('blur_strength', float, "If small (< 10), the filter will not have much effect, if large (> 150),"
                                         "will have a very strong effect, making the image look 'cartoonish'", 80),
                ("filter_size", int, "Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications,"
                                     " and perhaps d=9 for offline applications that need heavy noise filtering.", 5)]

    def apply(self, image, kwargs):
        # magic number parameters for bilateralFilter can be modified if a better value is found, follow
        # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
        # for information on how they work. the test_images_filters.py script can also be useful for trying settings
        result = cv2.bilateralFilter(image, d=self.filter_size,
                                     sigmaColor=self.blur_strength,
                                     sigmaSpace=self.blur_strength)
        return result


class ImgMotionBlurRule(MetRule):
    """Applies a motion blur filter to the image, to replicate the real motion blur commonly found in videos.
        Motion blur can be simulated with a Point Spread Function, a kernel full of zeroes except for a line of ones,
        in the direction in which we want to simulate the motion blur.
        """
    def __init__(self, **kwargs):
        """Motion Blur Metamorphic Rule.

        Applies a motion blur filter to the image, to replicate the real motion blur commonly found in videos. A human
        manual process will be done to confirm that the images that naturally have motion blur in the dataset present
        the same pose estimation degradation as the ones where we apply our motionBlur filter.

        Motion blur can be simulated with a Point Spread Function, a kernel full of zeroes except for a line of ones,
        in the direction in which we want to simulate the motion blur. The size of the kernel will determine the
        strength of the blur. There is then two parameters, blur strength or length, and blur orientation or angle.

        :param kwargs:
            :param blur_strength: should be odd, since the kernel needs a center point to rotate around
            :param orientation: the angle is measured in degrees counter-clockwise from the horizontal axis
        """
        super().__init__(**kwargs)
        self.size = kwargs['kernel_size']
        self.angle = kwargs['orientation']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('kernel_size', int, "the greater the kernel size, the bigger the blur effect."
                                     "should be odd, since the kernel needs a center point to rotate around", 7),
                ('orientation', int, "the angle is measured in degrees counter-clockwise from the horizontal axis", 0)]

    def apply(self, image, kwargs):
        # I could not find the original link where I found the idea for this code. This is the closes I could find when looking now:
        # https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array https://www.iditect.com/programming/python-example/opencv-motion-blur-in-python.html

        PSF_kernel = np.zeros((self.size, self.size))
        # Fill the middle column of the kernel with ones
        PSF_kernel[int((self.size - 1) / 2), :] = np.ones(self.size)
        # Normalize the kernel so we don't make the image brighter
        PSF_kernel /= self.size

        # Use a rotation matrix to rotate the above kernel
        rotation_matrix = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), self.angle, 1)
        PSF_kernel = cv2.warpAffine(PSF_kernel, rotation_matrix, (self.size, self.size))

        # Apply the kernel to the image
        # ddepth = -1 means the output image will have the same depth as the input image
        result = cv2.filter2D(image, -1, PSF_kernel)
        return result


class ImgMirrorRule(MetRule):
    """ This rule mirrors an input image by flipping it from left to right, up and down, or both.
        """

    def __init__(self, **kwargs):
        """Mirror Metamorphic Rule.

        This rule mirrors an input image by flipping it from left to right (flip_code>0),
        up and down (flip_code==0), or both (flip_code<0).
        https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441

        :param kwargs:
            :param flip_code:
        """
        super().__init__(**kwargs)
        self.flip_code = kwargs['flip_code']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [("flip_code", int,
                 "flip image from left to right (flip_code>0) up and down (flip_code==0), or both (flip_code<0).", 1)]

    def apply(self, image, kwargs):
        """Mirror Metamorphic Rule.

        This rule mirrors an input image by flipping it from left to right (flip_code>0),
        up and down (flip_code==0), or both (flip_code<0).

        :param image:
        :return:
        """
        result = cv2.flip(image, flipCode=self.flip_code)
        return result

    def resize_keypoints(self, keypoints, orig_img, modif_img):
        """The mirror rule needs to mirror also the keypoints according to its flip_code"""
        assert orig_img.shape[0] == modif_img.shape[0]
        assert orig_img.shape[1] == modif_img.shape[1]

        def mirror_point(point):
            """each point is a nparray with values [x, y, z, visibility?], the visibility is only present
            in pose landmarks, not face or hands"""
            if self.flip_code > 0:  # flip horizontally
                point[0] = 1 - point[0]
            elif self.flip_code == 0:  # flip vertically
                point[1] = 1 - point[1]
            elif self.flip_code < 0:
                point[0] = 1 - point[0]
                point[1] = 1 - point[1]
            else:
                raise "Unknown flip_code to resize the keypoints"
            return point

        # flip each keypoint along the same axis that the image was flipped
        keypoints = {
            key: None if val is None else np.stack([mirror_point(l) for l in val])
            for key, val in keypoints.items()
        }

        # Only if the image was mirrored horizontally, we need to swap keypoints left and right
        # This is not mirroring them in space like we did before, it's switching up which index correspond to each point
        if self.flip_code > 0 or self.flip_code < 0:
            def swap_keypoints(array, idx_a, idx_b):
                """aux function to avoid human typos, we only have to write each idx once for the swap"""
                array[[idx_a, idx_b]] = array[[idx_b, idx_a]]

            if keypoints["pose"] is not None:
                # swap every left keypoint for its right counterpart. e.g. right shoulder <-> left shoulder
                pose = keypoints["pose"]
                # pose keypoints map https://1.bp.blogspot.com/-w22Iw7BRZsg/XzWx-S7DtpI/AAAAAAAAGZg/zgpN2e5Oye8qPXfq0zLq6dm38afXaUa8gCLcBGAsYHQ/s1999/image4%2B%25281%2529.jpg
                swap_keypoints(pose, 1, 4)  # swap inner eyes
                swap_keypoints(pose, 2, 5)  # swap eyes
                swap_keypoints(pose, 3, 6)  # swap outer eyes
                swap_keypoints(pose, 7, 8)  # swap ears
                swap_keypoints(pose, 9, 10)  # swap mouth
                swap_keypoints(pose, 11, 12)  # swap shoulders
                swap_keypoints(pose, 13, 14)  # swap elbows
                swap_keypoints(pose, 15, 16)  # swap wrists
                swap_keypoints(pose, 17, 18)  # swap pinky knuckles
                swap_keypoints(pose, 19, 20)  # swap index knuckles
                swap_keypoints(pose, 21, 22)  # swap thumbs knuckles
                swap_keypoints(pose, 23, 24)  # swap hips
                swap_keypoints(pose, 25, 26)  # swap knees
                swap_keypoints(pose, 27, 28)  # swap ankles
                swap_keypoints(pose, 29, 30)  # swap heels
                swap_keypoints(pose, 31, 32)  # swap foot index

            # the hand map is the same for l_hand and r_hand, we just need to switch which hand is witch
            # https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md#hand-landmark-model
            keypoints["left_hand"], keypoints["right_hand"] = keypoints["right_hand"], keypoints["left_hand"]

            def swap_all_keypoints_in_simetric_lists(array, list_a, list_b):
                for a,b in zip(list_a, list_b):
                    swap_keypoints(array, a, b)

            # Now we swap all the many many keypoints in the face
            if keypoints["face"] is not None:
                face = keypoints["face"]
                # face landmarks gotten from https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
                # https://github.com/google/mediapipe/issues/1615  and  https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
                middle_line = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
                               0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]  # this ones need no change
                right_face_silhuoette = [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                         397, 365, 379, 378, 400, 377]
                left_face_silhuoette = [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58,
                                        172, 136, 150, 149, 176, 148]
                swap_all_keypoints_in_simetric_lists(face, right_face_silhuoette, left_face_silhuoette)
                right_middle_silhuoette = [337, 299, 333, 298, 301, 368, 264, 447, 366, 401,
                                           435, 367, 364, 394, 395, 369, 396]
                left_middle_silhuoette = [108, 69, 104, 68, 71, 139, 34, 227, 137, 177, 215,
                                          138, 135, 169, 170, 140, 171]
                swap_all_keypoints_in_simetric_lists(face, right_middle_silhuoette, left_middle_silhuoette)
                right_inner_silhuoette = [336, 296, 334, 293, 300, 383, 372, 345, 352, 376,
                                          433, 416, 434, 430, 431, 262, 428]
                left_inner_silhuoette = [107, 66, 105, 63, 70, 156, 143, 116, 123, 147, 213,
                                         192, 214, 210, 211, 32, 208]
                swap_all_keypoints_in_simetric_lists(face, right_inner_silhuoette, left_inner_silhuoette)

                right_outer_inface = [351, 412, 343, 277, 329, 330, 280, 411, 427, 436, 432,
                                      422, 424, 418, 421]
                left_outer_inface = [122, 188, 114, 47, 100, 101, 50, 187, 207, 216, 212, 202,
                                     204, 194, 201]
                swap_all_keypoints_in_simetric_lists(face, right_outer_inface, left_outer_inface)
                right_inner_inface = [419, 399, 437, 355, 371, 266, 425, 426, 322, 410, 287, 273,
                                      335, 406, 313]
                left_inner_inface = [196, 174, 217, 126, 142, 36, 205, 206, 92, 186, 57, 43,
                                     106, 182, 83]
                swap_all_keypoints_in_simetric_lists(face, right_inner_inface, left_inner_inface)

                right_outer_nose = [248, 456, 420, 429, 358, 423, 391, 393]
                left_outer_nose = [3, 236, 198, 209, 129, 203, 165, 167]
                swap_all_keypoints_in_simetric_lists(face, right_outer_nose, left_outer_nose)
                right_middle_nose = [281, 363, 360, 279, 331, 294, 327, 326]
                left_middle_nose = [51, 134, 131, 49, 102, 64, 98, 97]
                swap_all_keypoints_in_simetric_lists(face, right_middle_nose, left_middle_nose)
                outer_right_nostril = [275, 440, 344, 278, 439, 455, 460, 328]
                outer_left_nostril = [45, 220, 115, 48, 219, 235, 240, 99]
                swap_all_keypoints_in_simetric_lists(face, outer_right_nostril, outer_left_nostril)
                middle_right_nostril = [274, 457, 438, 392, 289, 305, 290]
                middle_left_nostril = [44, 237, 218, 166, 59, 75, 60]
                swap_all_keypoints_in_simetric_lists(face, middle_right_nostril, middle_left_nostril)
                inner_right_nostril = [354, 461, 458, 459, 309, 250, 462, 370]
                inner_left_nostril = [125, 241, 238, 239, 79, 20, 242, 141]
                swap_all_keypoints_in_simetric_lists(face, inner_right_nostril, inner_left_nostril)

                # the lips landmarks in these lists are from left to right, and in the middle have the middle line.
                # so to swap them we do the reversed iteration of the second half of the list
                lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
                swap_all_keypoints_in_simetric_lists(face, lipsUpperOuter[:(len(lipsUpperOuter) - 1)//2],
                                                           lipsUpperOuter[:(len(lipsUpperOuter) - 1)//2:-1])
                lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375]
                swap_all_keypoints_in_simetric_lists(face, lipsLowerOuter[:(len(lipsLowerOuter) - 1) // 2],
                                                           lipsLowerOuter[:(len(lipsLowerOuter) - 1) // 2:-1])
                lips_middle_1_upper = [76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306]
                swap_all_keypoints_in_simetric_lists(face, lips_middle_1_upper[:(len(lips_middle_1_upper) - 1) // 2],
                                                           lips_middle_1_upper[:(len(lips_middle_1_upper) - 1) // 2:-1])
                lips_middle_2_upper = [62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292]
                swap_all_keypoints_in_simetric_lists(face, lips_middle_2_upper[:(len(lips_middle_2_upper) - 1) // 2],
                                                           lips_middle_2_upper[:(len(lips_middle_2_upper) - 1) // 2:-1])
                lips_middle_1_lower = [77, 90, 180, 85, 16, 315, 404, 320, 307]
                swap_all_keypoints_in_simetric_lists(face, lips_middle_1_lower[:(len(lips_middle_1_lower) - 1) // 2],
                                                           lips_middle_1_lower[:(len(lips_middle_1_lower) - 1) // 2:-1])
                lips_middle_2_lower = [96, 89, 179, 86, 15, 316, 403, 319, 325]
                swap_all_keypoints_in_simetric_lists(face, lips_middle_2_lower[:(len(lips_middle_2_lower) - 1) // 2],
                                                           lips_middle_2_lower[:(len(lips_middle_2_lower) - 1) // 2:-1])
                lips_upper_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
                swap_all_keypoints_in_simetric_lists(face, lips_upper_inner[:(len(lips_upper_inner) - 1) // 2],
                                                           lips_upper_inner[:(len(lips_upper_inner) - 1) // 2:-1])
                lips_lower_inner = [95, 88, 178, 87, 14, 317, 402, 318, 324]
                swap_all_keypoints_in_simetric_lists(face, lips_lower_inner[:(len(lips_lower_inner) - 1) // 2],
                                                           lips_lower_inner[:(len(lips_lower_inner) - 1) // 2:-1])

                rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
                rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
                rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190]
                rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
                rightEyeUpper2 = [113, 225, 224, 223, 222, 221, 189]
                rightEyeLower2 = [226, 31, 228, 229, 230, 231, 232, 233, 244]
                rightEyebrowLower = [35, 124, 46, 53, 52, 65, 55, 193]
                rightEyeLower3 = [111, 117, 118, 119, 120, 121, 128, 245]
                rightEyeIris = [473, 474, 475, 476, 477]

                leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
                leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
                leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
                leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
                leftEyeUpper2 = [342, 445, 444, 443, 442, 441, 413]
                leftEyeLower2 = [446, 261, 448, 449, 450, 451, 452, 453, 464]
                leftEyeLower3 = [340, 346, 347, 348, 349, 350, 357, 465]
                leftEyebrowLower = [265, 353, 276, 283, 282, 295, 285, 417]
                leftEyeIris = [468, 469, 470, 471, 472]

                swap_all_keypoints_in_simetric_lists(face, rightEyeUpper0, leftEyeUpper0)
                swap_all_keypoints_in_simetric_lists(face, rightEyeLower0, leftEyeLower0)
                swap_all_keypoints_in_simetric_lists(face, rightEyeUpper1, leftEyeUpper1)
                swap_all_keypoints_in_simetric_lists(face, rightEyeLower1, leftEyeLower1)
                swap_all_keypoints_in_simetric_lists(face, rightEyeUpper2, leftEyeUpper2)
                swap_all_keypoints_in_simetric_lists(face, rightEyeLower2, leftEyeLower2)
                swap_all_keypoints_in_simetric_lists(face, rightEyeLower3, leftEyeLower3)
                swap_all_keypoints_in_simetric_lists(face, rightEyebrowLower, leftEyebrowLower)
                # swap_all_keypoints_in_simetric_lists(face, rightEyeIris, leftEyeIris)  our current mediapipe model does not include the irises

        return keypoints


class ImgBlackWhiteRule(MetRule):
    """This rule changes the color channels of the image from RGB, to gray, and then back to RGB. Maintaining the
    color channels but losing all color information and only leaving the black and white value
    """
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return []

    def apply(self, image, kwargs):
        """Convert RGB image to black and white using cv2.
        Reconvert it to RGB format so the rest of the code works correctly, but the color information will be lost

        :param image:
        :return:
        """
        result = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
        result = cv2.cvtColor(result, code=cv2.COLOR_GRAY2RGB)
        return result


class ImgDarkenBrightenRule(MetRule):
    """Make the image darker or brighter in one of 3 ways according to "method" argument. Either add a constant value,
       multiply each pixel by a constant, or use gamma correction.
            """
    def __init__(self, **kwargs):
        """Image Brightness change Metamorphic Rule.

        Make the image darker or brighter in one of 3 ways according to "method" argument. Either add a constant value,
        multiply each pixel by a constant, or use gamma correction. Following
        https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

        :param kwargs:
            :param brightness_constant: a constant to add to each pixel, linear brightening, will make img flatter,
                    only used if method is scale
            :param brightness_multiplier: constant to multiply each pixel by, keeps image contrast,
                    only used if method is scale
            :param gamma: only used if method is gamma
            :param method: choose between "gamma", "scale"
        """
        super().__init__(**kwargs)
        self.brightness_constant = kwargs['brightness_constant']
        self.brightness_multiplier = kwargs['brightness_multiplier']
        self.gamma = kwargs['gamma']
        self.method = kwargs['method']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('brightness_constant', int,
                 """a constant to add to each pixel, linear brightening, will make img flatter,
                  only used if method is scale""", 20),
                ('brightness_multiplier', float, """constant to multiply each pixel by, keeps image contrast,
                    only used if method is scale""", 1.2),
                ('gamma', float, "only used if method is gamma", 1.2),
                ('method', str, "choose between 'gamma', 'scale'", "gamma")]

    def apply(self, image, kwargs):
        """Make the image brighter or darker

        :param image:
        :return:
        """
        match self.method:
            case "gamma":
                lookUpTable = np.empty((1, 256), np.uint8)
                for i in range(256):
                    lookUpTable[0, i] = np.clip(pow(i / 255.0, self.gamma) * 255.0, 0, 255)
                result = cv2.LUT(image, lookUpTable)
            case "scale":
                result = cv2.convertScaleAbs(image, alpha=self.brightness_multiplier, beta=self.brightness_constant)
            case _:
                raise "please select a method for brightening the image from those provided"
        return result


class ImgRotationRule(MetRule):
    """Rotate the image around any point in the image.
    """
    def __init__(self, **kwargs):
        """Image Rotation Metamorphic Rule.

        The idea of choosing a different center is to choose where the "black pixels" will be where we have no
        information. This can help for example to be more realistic since if we rotate around a lower corner, we won't
        have a lack of information under the waist of a human, but simply a dark "roof" over their head

        :param kwargs:
            :param rotation_angle: degrees to rotate the image by, positive values mean counter-clockwise rotation
            :param center: should be a point between (0, 0) and (1, 1), showing were in the image to rotate around
        """
        super().__init__(**kwargs)
        self.rotation_angle = kwargs['rotation_angle']
        self.center = kwargs['center']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """

        "this function is used by argparser to know how to interpret the argument that is always inputed as a string"
        def tuple_type(input_string: str):
            try:
                x, y = map(float, input_string.strip("()").split(','))
                return x, y
            except:
                raise argparse.ArgumentTypeError("Input should be in format x,y where x and y are floats")

        return [('rotation_angle', int, "how much to rotate the image by (in degrees)", 15),
                ('center', tuple_type,
                 "should be a point between (0, 0) and (1, 1), showing were in the image to rotate around",
                 "(0.5, 0.5)")]

    def apply(self, image, kwargs):
        """Return the rotated image

        :param image:
        :return:
        """

        center = image.shape[0] * self.center[0], image.shape[1] * self.center[1]
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1)
        result = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return result

    def resize_keypoints(self, keypoints, orig_img, modif_img):
        """The rotation rule needs to rotate the keypoints back to reverse the modification of the image"""
        assert orig_img.shape[0] == modif_img.shape[0]
        assert orig_img.shape[1] == modif_img.shape[1]

        def rotate(origin, point, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            gotten from https://stackoverflow.com/a/34374437
            """
            ox, oy = origin
            px, py = point

            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy

        def unrotate_point(point):
            """each point is a nparray with values [x, y, z, visibility?], the visibility is only present
            in pose landmarks, not face or hands"""
            rotation_radians = math.radians(self.rotation_angle)
            point_x, point_y = rotate(self.center, (point[0], point[1]), rotation_radians)
            point[0] = point_x
            point[1] = point_y
            return point

        # flip each keypoint along the same axis that the image was flipped
        keypoints = {
            key: None if val is None else np.stack([unrotate_point(l) for l in val])
            for key, val in keypoints.items()
        }

        return keypoints


class ImgResolutionRule(MetRule):
    """reduce the image pixel resolution to simulate a person/object farther away from the camera
     or a bad quality photo.
     The img resolution rule needs no resizing for keypoints since they are normalized between 0 and 1 and the
    relative position of things inside the image doesn't change when changing the resolution"""
    def __init__(self, **kwargs):
        """Image Resolution Reduction Metamorphic Rule.

        reduces the pixel size of an image based on the size_ratio * original_size

        :param kwargs:
            :param size_ratio:
        """
        super().__init__(**kwargs)
        self.size_ratio = kwargs['size_ratio']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('size_ratio', float, "modified resolution will be (orig_width*size_ratio, orig_height*size_ratio)", 0.6)]

    def apply(self, image, kwargs):
        """Reduce the resolution of an image

        We assume this class will be used to reduce the size of an image (size_ratio < 1), so based on
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
        we use INTER_AREA for reducing the size instead of the default interpolation which is for increasing

        :param image:
        :return:
        """
        result = cv2.resize(image, (0, 0), fx=self.size_ratio, fy=self.size_ratio, interpolation=cv2.INTER_AREA)
        return result


class ImgStretchRule(MetRule):
    """Similar to ImgResolutionRule, but uses a different ratio for height and width, the idea is to change the
    proportions of the image, not reduce the resolution.

    The img stretch rule needs no resizing for keypoints since they are normalized between 0 and 1 and the
    relative position of things inside the image doesn't change when stretching the image
    """
    def __init__(self, **kwargs):
        """Image Stretching Deformation Metamorphic Rule.

        Change the shape of the image based on the height_ratio and width_ratio

        :param kwargs:
            :param height_ratio:
            :param width_ratio:
        """
        super().__init__(**kwargs)
        self.height_ratio = kwargs['height_ratio']
        self.width_ratio = kwargs['width_ratio']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('height_ratio', float, "how much to stretch/shrink height", 1.15),
                ('width_ratio', float, "how much to stretch/shrink width", 1)]

    def apply(self, image, kwargs):
        """Change the shape of an image

        :param image:
        :return:
        """
        result = cv2.resize(image, (0, 0), fx=self.width_ratio, fy=self.height_ratio)
        return result


class AuxSegmentImageMask:
    """used to save an intermediate modified version of the images to mask the clothes/skin/background of the images,
    and allow more advanced metamorphic rules."""
    def __init__(self):
        """initialize the mediapipe segmenter only once to optimize runtime"""

        model_path = 'Datasets/mediapipe_multiclass_segmenter.tflite'
        # Create an image segmenter instance with the image mode:
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            output_category_mask=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE, )
        self.mediapipe_segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    @staticmethod
    def kmeans_to_thresholds(image, k_clusters, rounds=1):
        """use opencv kmeans to segment the image into parts, previous to finding the mask.
            Unused for now, just as a backup to compare against mediapipe segmentation

            Param k_clusters is the number of sections in which we want to divide the image
        """
        # https://stackoverflow.com/questions/60272082/how-to-segment-similar-looking-areas-color-wise-inside-a-image-that-belong-to
        # https://docs.opencv.org/4.1.1/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
        h, w = image.shape[:2]
        samples = np.zeros([h * w, 3], dtype=np.float32)
        count = 0

        for x in range(h):
            for y in range(w):
                samples[count] = image[x][y]
                count += 1

        compactness, labels, centers = cv2.kmeans(samples,
                                                  k_clusters,
                                                  None,
                                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                                  rounds,
                                                  cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        return res.reshape(image.shape)

    @staticmethod
    def find_mask(image, kind):
        """kind is either clothes or skin or background
        Unused for now, just as a backup to compare against mediapipe segmentation"""
        # https://medium.com/srm-mic/color-segmentation-using-opencv-93efa7ac93e2

        # first blur the image?
        blur = cv2.blur(image, (5, 5))
        blur0 = cv2.medianBlur(blur, 5)
        blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
        blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)

        # convert to hsv since it helps with color segmentation
        hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

        # use the correct color thresholds
        color_thresholds = {"example_bird": (np.array([55, 0, 0]), np.array([118, 255, 255]))}
        low_color_thresh, high_color_thresh = color_thresholds[kind]
        mask = cv2.inRange(hsv, low_color_thresh, high_color_thresh)

        # the mask can be used in this way
        # res = cv.bitwise_and(img, img, mask=mask)
        return mask

    def mediapipe_segmentation(self, image):
        """returns numpy 2D array of the same size as the image, in each pixel position, it has an integer between
        0 and 5, detailing which class is estimated to be there:
            0 - background
            1 - hair
            2 - body-skin
            3 - face-skin
            4 - clothes
            5 - others (accessories)"""

        # image needs to be adapted to mediapipe format for the uses from mp.tasks, like the segmenter
        # it is not clear from the docs if the image that goes into mp.Image should be BGR or RGB,
        # but the results seem similar so for now we use RGB based on this issue
        # https://github.com/google/mediapipe/issues/5218
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Perform image segmentation on the provided single image.
        # The image segmenter must be created with the image mode.
        segmentation_result = self.mediapipe_segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

        categories = category_mask.numpy_view()

        return categories

    def get_mask(self, image, mask_type, kwargs):
        """for now, we always use mediapipe segmentation, old ad_hoc implementations using kmeans and color thresholding
        can be used as alternatives to compare in the future

        :param image:
        :param mask_type: what mask to create, from "skin", "clothes", "hair", "background"
        :param kwargs:
            :param image_filepath: the path to the original image, needed to save the mask in a new filepath
            :param data_type: what dataset folder is the image in, between dev, test, train, etc.
        """
        image_filepath = kwargs['image_filepath']
        data_type = kwargs['data_type']
        dataset_name = kwargs['dataset_name']

        mask_filepath = get_modified_img_path(image_filepath, "segment_categories", data_type, dataset_name)

        # if it has been already saved, don't compute it again
        if os.path.isfile(mask_filepath):
            categories = cv2.imread(mask_filepath)
        else:
            categories = self.mediapipe_segmentation(image)
            categories = cv2.merge((categories, categories, categories))  # mask needs to be 3 channels instead of 1
            # if save_modified_images:  in case we ever want to not save the segment categories
            save_modified_image(categories, mask_filepath)

        relevant_categories = {
            "skin": [2, 3],
            "clothes": [4],
            "hair": [1],
            "background": [0]
        }
        # leave a 1 in the pixels where the desired category was located
        mask = np.isin(categories, relevant_categories[mask_type])
        mask = mask.astype(np.uint8)
        mask *= 255
        return mask




class ImgSilhouetteRule(MetRule):
    """It should simply leave the silhouette of different segments of the image according to the mask rule, to
    completely remove textures and colors. This is equivalent to just filling in the image with a constant color value,
    which then the maskRule can properly segment."""
    def __init__(self, **kwargs):
        """Image Silhouette Metamorphic Rule.

        This rule is intended to always be used with the "masked" Rule, this way, the segmented part of the image, with
        the silhouette of the person/clothes/object can be replaced with the same shape but a constant color

        :param kwargs:
            :param color_constant: the RGB triple constant color value to apply to each pixel
        """
        super().__init__(**kwargs)
        self.color_constant = kwargs['color_constant']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        def color_tuple_type(input_string: str):
            try:
                x, y, z = map(float, input_string.strip("()").split(','))
                return x, y, z
            except:
                raise argparse.ArgumentTypeError("Input should be like (x,y,z) where x, y and z are ints in [0,255]")

        return [('color_constant', color_tuple_type, "color of the plain image to return", (0, 0, 0))]

    def apply(self, image, kwargs):
        result = image.copy()
        result[:] = self.color_constant
        return result


class ImgColorWheelRule(MetRule):
    """first find the average skin color of the people in a very homogeneous dataset with simple backgrounds,
        like https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/, then change this and colors in the image
        to other arbitrary colors, to ensure that the DL models recognize the shape of humans, and not some colors in
        particular.

        https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html explains hsv in opencv, detailing how to find
        the color thresholds in hsv that we care about
        """
    def __init__(self, **kwargs):
        self.hue_rotation = kwargs['hue_rotation']
        pass

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('hue_rotation', int, "hue wheel rotation angle, int between 0 and 180", 90)]

    def apply(self, image, kwargs):
        result = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(result)

        # in opencv hue range is [0, 179] instead of [0, 255] like for saturation and value
        hue = ((hue.astype(np.uint16) + self.hue_rotation) % 180).astype(np.uint8)

        result = cv2.merge([hue, saturation, value])
        result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
        return result


class ImgColorChannelsRule(MetRule):
    """"""
    def __init__(self, **kwargs):
        """Image Color Channels Metamorphic Rule.

        We will manipulate the color channels to test how much the models depend on the colors of the images instead
        of the shapes to identify things. We will either eliminate all information from one color channel, or make one
        more prevalent, or change the channels meaning, etc. The input images are always in RGB

        :param kwargs:
            :param color_channels_out: a string describing the output format of the image, (RGB, BGR, GRAY, etc.),
            it needs to match a cv2 color option like cv2.COLOR_RGB2{color_channels_out}.
                Note that the model expects an RGB image, so this is just to give different colors to the model
                see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html for color conversions options,
                for example, using grayscale and multipliers (0, 1, 0) will give a green range image
            :param channel_multipliers: for every original pixel (r, g, b) we would return
                (r*channel_multipliers[0], g*channel_multipliers[1], b*channel_multipliers[2]) capped between [0, 255]
        """
        super().__init__()
        self.color_channels_out = kwargs['color_channels_out']
        self.channel_multipliers = kwargs['channel_multipliers']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        def color_tuple_type(input_string: str):
            try:
                x, y, z = map(float, input_string.strip("()").split(','))
                return x, y, z
            except:
                raise argparse.ArgumentTypeError("Input should be like (x,y,z) where x, y and z are floats")

        return [('color_channels_out', str, "string describing the format of the output image", "RGB"),
                ('channel_multipliers', color_tuple_type, "how much to multiply each channel by", (1, 1, 1))]

    def apply(self, image, kwargs):
        if self.color_channels_out != "RGB":
            output_format = getattr(cv2, "COLOR_RGB2"+self.color_channels_out)
            result = cv2.cvtColor(image, output_format)
            if output_format == cv2.COLOR_RGB2GRAY:
                result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # image needs to have 3 channels again
        else:
            result = image.copy()

        color_channels = list(cv2.split(result))
        for color in range(0, 3):
            color_channels[color] = (color_channels[color] * self.channel_multipliers[color]).astype(np.uint8)
            color_channels[color] = np.clip(color_channels[color], 0, 255)
        result = cv2.merge(color_channels)
        return result

