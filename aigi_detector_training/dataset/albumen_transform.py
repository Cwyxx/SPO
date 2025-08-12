import albumentations as A
RESIZE_SIZE = (256, 256)
CROP_SIZE = (224, 224)
# real
train_real_transform = A.Compose([
    A.SmallestMaxSize(max_size=RESIZE_SIZE[0], p=1.0),
    A.RandomCrop(height=CROP_SIZE[0], width=CROP_SIZE[1], p=1.0),
    A.SquareSymmetry(p=0.5), # Replaces Horizontal/Vertical Flips
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    A.ToTensorV2(),
])
# fake
train_fake_transform = A.Compose([
    # JPEG compress
    A.ImageCompression(quality_range=(60, 100), p=1.0),
    A.SmallestMaxSize(max_size=RESIZE_SIZE[0], p=1.0),
    A.RandomCrop(height=CROP_SIZE[0], width=CROP_SIZE[1], p=1.0),
    A.SquareSymmetry(p=0.5), # Replaces Horizontal/Vertical Flips
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    A.ToTensorV2(),
])

val_real_transform = A.Compose([
    A.SmallestMaxSize(max_size=RESIZE_SIZE[0], p=1.0),
    A.CenterCrop(height=CROP_SIZE[0], width=CROP_SIZE[1], p=1.0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    A.ToTensorV2()
])

val_fake_transform = A.Compose([
    # JPEG compress
    A.ImageCompression(quality_range=(60, 100), p=1.0),
    A.SmallestMaxSize(max_size=RESIZE_SIZE[0], p=1.0),
    A.CenterCrop(height=CROP_SIZE[0], width=CROP_SIZE[1], p=1.0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    A.ToTensorV2(),
])