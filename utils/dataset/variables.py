SUPPORTED_DATASET_FORMAT = [
    "VOC",
]
BASE_SUPPORTED_IMAGE_TYPES = [
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tif",
    "tiff",
]
SUPPORTED_IMAGE_TYPES = [
    *BASE_SUPPORTED_IMAGE_TYPES,
    *[str(t).upper() for t in BASE_SUPPORTED_IMAGE_TYPES],
]
