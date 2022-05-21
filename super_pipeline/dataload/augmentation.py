from torchvision import transforms
from super_pipeline.main import lib


def trans01(trans_type):
    """
    Args:
    trans_type - transformation for data:
        "train" - transformation for train dataset,
        "val"   - transformation for valuation dataset
        "test"  - transformation for test dataset
    """
    if trans_type == "train":
        t_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            # transforms.RandomRotation(degrees=(0, 359)),
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
            # transforms.CenterCrop(img_size),
            transforms.Lambda(lib.img_normalize)
        ])
# torchvision.transforms.Lambda(lambd)
#     *[transforms.RandomResizedCrop(224)  for _ in range(1)],
#     transforms.Resize((224, 224)),
#     transforms.Resize(size=random.randint(50,224)), # 0.7190860215053764 ep 50
#     transforms.RandomPosterize(bits=3),             # 0.7204301075268817 ep 50
#     transforms.RandomInvert(),                      # 0.7298387096774194 ep 50
#     transforms.RandomHorizontalFlip(),              # 0.7311827956989247 ep 50
#     transforms.RandomResizedCrop(224),              # 0.7661290322580645 ep 50
# transforms.RandAugment(2),
# transforms.RandomHorizontalFlip(p=0.5),
# transforms.RandomVerticalFlip(p=0.5),
# transforms.TrivialAugmentWide(),
# transforms.RandAugment(),
# transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
# transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),

    elif trans_type in ["val", "test"]:
        t_transforms = transforms.Compose([
            # transforms.CenterCrop(img_size),
            transforms.Resize((512, 512)),
            transforms.Lambda(lib.img_normalize)
        ])
    else:
        print('Transform type {trans_type} must be "train", "val" or "test"')
    return t_transforms


