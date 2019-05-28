from .psp_dsn_resnet import get_psp_dsn_resnet101
from .student_res18 import get_psp_resnet18
from .student_res18_pre import get_psp_resnet18_pre
from .teacher_dense import get_teacher_dense
from .ESPnet import get_ESPnet_decoder, get_ESPnet_encoder

networks = {
    'psp_dsn': get_psp_dsn_resnet101,
    'student_res18': get_psp_resnet18,
    'student_res18_pre': get_psp_resnet18_pre,
    'teacher_dense': get_teacher_dense,
    'student_esp_e': get_ESPnet_encoder,
    'student_esp_d': get_ESPnet_decoder,
}

#
def get_segmentation_model(name, **kwargs):
    return networks[name.lower()](**kwargs)
