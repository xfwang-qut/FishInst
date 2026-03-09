from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class FishDataset(CocoDataset):
    """dataset for Cityscapes."""

    METAINFO = {
        'classes': ['Fish'],
        'palette': [(220, 20, 60)]


    }


@DATASETS.register_module()
class UW10KDataset(CocoDataset):
    """dataset for Cityscapes."""

    METAINFO = {
        'classes': ['Fish', 'Sea urchins', 'Sea cucumber', 'Sea turtle', 'Sea snake', 'Squid',
                    'Octopus', 'Shrimp', 'Ray', 'Shellfish', 'Seahorse', 'Starfish', 'Jellyfish',
                    'Diver', 'Coral'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                    (0, 80, 100), (0, 0, 230), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                    (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (0, 0, 230)]


    }


@DATASETS.register_module()
class USIS10KDataset(CocoDataset):
    """dataset for Cityscapes."""

    METAINFO = {
        'classes': ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants',
                    'human divers', 'robots', 'sea-floor'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230)]
    }


@DATASETS.register_module()
class UIISDataset(CocoDataset):
    METAINFO = {
        'classes': ['fish', 'reefs', 'aquatic plants', 'wrecks/ruins', 'human divers', 'robots', 'sea-floor'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230)]
    }


@DATASETS.register_module()
class UIIS10KDataset(CocoDataset):
    METAINFO = {
        'classes': ['fish', 'reptiles', 'arthropoda', 'corals', 'mollusk', 'plants', 'ruins', 'garbage', 'human', 'robots'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                    (0, 80, 100), (0, 0, 230), (220, 20, 60), (255, 0, 0), (0, 0, 142),]
    }