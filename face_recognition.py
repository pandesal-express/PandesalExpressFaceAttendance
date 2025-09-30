from typing import Union

import numpy
from PIL import Image
# from DeepFace module
from deepface.modules import modeling, detection, preprocessing
from deepface.models.FacialRecognition import FacialRecognition

import cv2
import numpy as np


def _auto_exposure(
    image_np,
    alpha=1.2,
    beta=10,
    saturation=25,
    target_brightness=122,
    shadow_percentile=5,
    highlight_percentile=95
) -> np.ndarray:
    """
    Auto-exposure with histogram-based balancing of shadows and highlights.

    Parameters:
        image_np (numpy.ndarray): Input image in RGB format.
        alpha (float): Default contrast multiplier.
        beta (int): Default brightness offset.
        saturation (int): Default saturation adjustment.
        target_brightness (int): Target median brightness (0-255).
        shadow_percentile (int): Percentile for the darkest regions.
        highlight_percentile (int): Percentile for the brightest regions.

    Returns:
        numpy.ndarray: Balanced image.
    """
    gray = cv2.cvtColor(src=image_np, code=cv2.COLOR_RGB2GRAY)

    # calculates the histogram of pixel intensities
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]

    # calculates the shadow and highlight thresholds
    shadow_threshold = np.searchsorted(cdf_normalized, shadow_percentile / 100.0)
    highlight_threshold = np.searchsorted(cdf_normalized, highlight_percentile / 100.0)

    # contrast stretching
    stretched = np.clip((gray - shadow_threshold) * (255 / (highlight_threshold - shadow_threshold)), 0, 255).astype(np.uint8)

    # this adjusts the brightness to match target brightness
    mean_brightness = np.mean(stretched)
    brightness_correction = target_brightness - mean_brightness
    adjusted_image = cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta + brightness_correction)

    hsv_image = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    # apply saturation channel
    s = cv2.add(s, saturation)

    balanced_image = cv2.merge((h, s, v))
    balanced_image = cv2.cvtColor(balanced_image, cv2.COLOR_HSV2RGB)

    return balanced_image


def embedding(
        image_path: Union[str, np.ndarray],
        model_name: str = "Facenet512",
        enforce_detection: bool = True,
        detector_backend: str = "opencv",
        align: bool = True,
        expand_percentage: int = 0,
        normalization: str = "base",
        anti_spoofing: bool = True,
):
    """
    Extracts multidimensional vector embeddings from the faces in the image.

    Improving the image quality first using auto exposure with histogram-based balancing of shadows and highlights, and then face encoding.

    Reference: @deepface
    Args:
        image_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face. Will convert to numpy array if it is a string.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet
            (default is Facenet512.).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set too False to avoid the exception for low-resolution images
            (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
            (default is base).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is True).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - embedding (List[float]): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).

        - facial_area (dict): Detected facial area by face detection in dictionary format.
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.

        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.

        - is_real (bool): Flag to indicate if the face is real or spoofed. If `anti_spoofing` is set.
    """
    resp_objs = []

    model: FacialRecognition = modeling.build_model(
        task="facial_recognition",
        model_name=model_name
    )

    target_size = model.input_shape
    img = image_path

    if isinstance(img, str):
        pil_image = Image.open(img)
        img = numpy.array(pil_image)

    balanced_image = _auto_exposure(image_np=img)

    img_objs = detection.extract_faces(
        img_path=balanced_image,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    for img_obj in img_objs:
        img = img_obj["face"]

        # rgb to bgr
        img = img[:, :, ::-1]

        confidence = img_obj["confidence"]

        # resize to expected shape of ml model
        img = preprocessing.resize_image(
            img=img,
            target_size=(target_size[1], target_size[0]),
        )

        # custom normalization
        img = preprocessing.normalize_input(img=img, normalization=normalization)

        vectors = model.forward(img)

        resp_objs.append(
            {
                "embedding": vectors,
                "face_confidence": confidence,
            }
        )

        if anti_spoofing:
            resp_objs[-1]["is_real"] = img_obj.get("is_real", True)

    return resp_objs
