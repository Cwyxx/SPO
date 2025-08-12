import os
import torch
from safetensors.torch import load_file
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

from univfd import UnivFD
from dataset.synthbuster import Synthbuster
from dataset.albumen_transform import val_real_transform, val_fake_transform

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test a UnivFD model.")
    parser.add_argument('--test_dataset_name', type=str, default="x-aigd/test")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--real_image_dir', type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4 )
    args = parser.parse_args()
    
    device = torch.device("cuda")
    model = UnivFD("openai/clip-vit-large-patch14")
    state_dict = load_file(args.ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    if args.test_dataset_name == "synthbuster":
        testset = Synthbuster(
            None,
            None,
            real_transform=val_real_transform,
            fake_transform=val_fake_transform
        )
        dataloader = torch.utils.data.DataLoader(
            testset, batch_size=4, num_workers=4, shuffle=False, drop_last=False
        )
        
    total_correct_fake = 0
    total_samples_fake = 0
    total_correct_real = 0
    total_samples_real = 0
    
    generator_stats = defaultdict(lambda: {'total_correct_fake': 0, 'total_samples_fake': 0})
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for uids, generators, image_paths, images, labels in tqdm(dataloader, desc=f"{args.test_dataset_name}"):
            images, labels = images.to(device), labels.to(device)
            predict_detection = model(images)
            
            # --- Accuracy Calculation ---
            predict_probabilities = torch.sigmoid(predict_detection)
            pred_det_labels = (predict_probabilities > 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predict_probabilities.cpu().numpy())
            
            real_index = (labels == 0)
            if real_index.sum() > 0:
                correct_real_preds = (pred_det_labels[real_index] == 0).sum().item()
                total_correct_real += correct_real_preds
                total_samples_real += real_index.sum().item()
                    
            fake_index = (labels == 1)
            if fake_index.sum() > 0:
                correct_fake_preds = (pred_det_labels[fake_index] == 1).sum().item()
                total_correct_fake += correct_fake_preds
                total_samples_fake += fake_index.sum().item()
                
                fake_indices = fake_index.nonzero().squeeze(1).tolist()
                fake_gens = [generators[idx] for idx in fake_indices]
                correct_fake_per_item = (pred_det_labels[fake_index] == 1).long().tolist()
                
                # --- Per-generator accuracy ---
                for i in range(len(labels)):
                    generator = generators[i]
                    label = labels[i].item()
                    pred = pred_det_labels[i].item()
                    
                    if label == 1: # fake
                        generator_stats[generator]['total_samples_fake'] += 1
                        if pred == 1:
                            generator_stats[generator]['total_correct_fake'] += 1
                
    
    real_accuracy = total_correct_real / (total_samples_real + 1e-9)
    fake_accuracy = total_correct_fake / (total_samples_fake + 1e-9)
    avg_accuracy = (real_accuracy + fake_accuracy) / 2
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    average_precision = average_precision_score(all_labels, all_predictions)
    
    print(f"model: {args.detect_method}")
    print(f"model_path: {args.model_path}")
    print(f"test_dataset_name: {args.test_dataset_name}")
    print(f"real accuracy: {real_accuracy*100:.2f}")
    print(f"fake accuracy: {fake_accuracy*100:.2f}")
    print(f"average accuracy: {avg_accuracy*100:.2f}")
    print(f"Average Precision (AP): {average_precision*100:.2f}%")
    
    print("\n--- Per-Generator Accuracy ---")
    for generator, stats in sorted(generator_stats.items()):
        g_fake_accuracy = stats['total_correct_fake'] / (stats['total_samples_fake'] + 1e-9)
        print(f"\nGenerator: {generator}")
        print(f"\tFake accuracy: {g_fake_accuracy*100:.2f}% ({stats['total_correct_fake']}/{stats['total_samples_fake']})")