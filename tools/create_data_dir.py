import os

def create_directory_structure(root_dir):
    # mmclassification structure
    mmclassification_structure = [
        "mmclassification/data/flowers/train",
        "mmclassification/data/flowers/test",
        "mmclassification/data/flowers/train_meta.list",
        "mmclassification/data/flowers/test_meta.list"
    ]

    # mmdetection structure
    mmdetection_structure = [
        "mmdetection/data/citypersons/annotations",
        "mmdetection/data/citypersons/leftImg8bit/train",
        "mmdetection/data/citypersons/leftImg8bit/val",
        "mmdetection/data/VOCdevkit/VOC2007",
        "mmdetection/data/VOCdevkit/VOC2012"
    ]

    # mmsegmentation structure
    mmsegmentation_structure = [
        "mmsegmentation/data/VOCdevkit/VOC2012/JPEGImages",
        "mmsegmentation/data/VOCdevkit/VOC2012/SegmentationClass",
        "mmsegmentation/data/VOCdevkit/VOC2012/ImageSets/Segmentation",
        "mmsegmentation/data/VOCdevkit/VOCaug/dataset/cls"
    ]

    # Combine all structures
    all_structure = mmclassification_structure + mmdetection_structure + mmsegmentation_structure

    # Create directories
    for directory in all_structure:
        dir_path = os.path.join(root_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    create_directory_structure(root_dir)
