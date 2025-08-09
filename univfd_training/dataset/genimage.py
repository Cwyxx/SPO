import torch
import os
import cv2

class Genimage(torch.utils.data.Dataset):
    def __init__(self, accelerator, image_dir, dataset_type, generator_list, real_transform=None, fake_transform=None):
        self.real_transform = real_transform
        self.fake_transform = fake_transform
        
        self.image_paths = []
        self.corresponding_generators = []
        self.image_labels = []
        
        for generator in generator_list:
            for _ in sorted(os.listdir(os.path.join(image_dir, generator, dataset_type, 'nature'))):
                self.image_paths.append(os.path.join(image_dir, generator, dataset_type, 'nature', _))
                self.corresponding_generators.append(generator)
                self.image_labels.append(0)
        real_image_num = len(self.image_paths)
        
        for generator in generator_list:
            for _ in sorted(os.listdir(os.path.join(image_dir, generator, dataset_type, 'ai'))):
                self.image_paths.append(os.path.join(image_dir, generator, dataset_type, 'ai', _))
                self.corresponding_generators.append(generator)
                self.image_labels.append(1)
        fake_image_num = len(self.image_paths) - real_image_num
        
        accelerator.print(f"dataset_type: {dataset_type}, real images: {real_image_num}, fake images: {fake_image_num}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        uid = os.path.splitext(os.path.basename(image_path))[0]
        generator = self.corresponding_generators[idx]
        label = self.image_labels[idx]
        
        try: 
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] Failed to read image {image_path}: {e}")
            exit(0)
       
        if label == 0 and self.real_transform:
            augmented = self.real_transform(image=image)
            image = augmented['image']
            
        elif label == 1 and self.fake_transform:
            augmented = self.fake_transform(image=image)
            image = augmented['image']

        return uid, generator, image_path, image, label