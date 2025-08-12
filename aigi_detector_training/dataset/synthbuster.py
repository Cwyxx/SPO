import torch
import os
import cv2

class Synthbuster(torch.utils.data.Dataset):
    def __init__(self, real_image_dir, fake_image_dir, real_transform=None, fake_transform=None):
        self.real_transform = real_transform
        self.fake_transform = fake_transform
        
        self.image_paths = []
        self.corresponding_generators = []
        self.image_labels = []
        
        if real_image_dir is not None and os.path.exists(real_image_dir):
            for _ in sorted(os.listdir(real_image_dir)):
                self.image_paths.append(os.path.join(real_image_dir, _))
                self.corresponding_generators.append('real')
                self.image_labels.append(0)
        real_image_num = len(self.image_paths)
        
        if fake_image_dir is not None and os.path.exists(fake_image_dir):
            generator_list = [ "dalle2", "dalle3", "firefly", "glide", "midjourney-v5", "stable-diffusion-1-3", "stable-diffusion-1-4", "stable-diffusion-2", "stable-diffusion-xl" ]
            # generator_list = ["stable-diffusion-1-4"]
            for generator in generator_list:
                for _ in os.listdir(os.path.join(fake_image_dir, generator)):
                    self.image_paths.append(os.path.join(fake_image_dir, generator, _))
                    self.corresponding_generators.append(generator)
                    self.image_labels.append(1)
                
        fake_image_num = len(self.image_paths) - real_image_num
        assert len(self.image_paths) == len(self.image_labels), "Image paths, generators, and labels must have the same length."
        print(f"Real images: {real_image_num}, Fake images: {fake_image_num}")
        
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