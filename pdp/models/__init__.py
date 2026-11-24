
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available


_import_structure = {
    "configuration_deformable_detr": ["DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeformableDetrConfig"],
}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_deformable_detr"] = ["DeformableDetrFeatureExtractor"]
    _import_structure["image_processing_deformable_detr"] = ["DeformableDetrImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_deformable_detr"] = [
        "DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DeformableDetrForObjectDetection",
        "DeformableDetrModel",
        "DeformableDetrPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_deformable_detr import DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DeformableDetrConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_deformable_detr import DeformableDetrFeatureExtractor
        from .image_processing_deformable_detr import DeformableDetrImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_deformable_detr import (
            DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            DeformableDetrForObjectDetection,
            DeformableDetrModel,
            DeformableDetrPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
