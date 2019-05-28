from .cityscapes import CitySegmentationDense,CitySegmentationLight,CitySegmentationSD,CitySegmentationLightesp,CitySegmentationesp,CitySegmentationunlabelesp,CamVidLight
from .Cityscapes_test import CitySegmentationLight_test,CitySegmentationesp_test
datasets = {
    'cityscapes_dense': CitySegmentationDense,
    'cityscapes_light': CitySegmentationLight,
    'cityscapes_light_test': CitySegmentationLight_test,
    'cityscapes_sd': CitySegmentationSD,
    'cityscapes_esp':CitySegmentationesp,
    'cityscapes_esp_test': CitySegmentationesp_test,
    'cityscapes_esp_sd': CitySegmentationLightesp,
    'cityscapes_esp_unlabel':CitySegmentationunlabelesp,
    'camvid_light':CamVidLight

}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)

# Conduct experiments on Mapillary_Vistas Dataset in our future work.
# https://github.com/ansleliu/LightNet/blob/master/datasets/mapillary_vistas_loader.py