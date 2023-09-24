from .dataset_example_ms import BraTS_new

datasets = {
    'new': BraTS_new
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
