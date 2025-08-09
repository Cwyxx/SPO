import os
import argparse
import torch
from accelerate import Accelerator
from univfd import UnivFD
from dataset.albumen_transform import train_real_transform, train_fake_transform, val_real_transform, val_fake_transform
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="univfd_training",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--tracker_experiment_name",
        type=str,
        default="genimage",
        help=(
            "The experiment name to be passed to Accelerator.init_trackers. "
            "This is used to identify the current experiment in tracking tools like swanlab. "
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoint",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--dataset_name", type=str, default="genimage")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(mixed_precision="no", log_with="swanlab")
    set_seed(1)
    weight_type = torch.float32
    
    # Initialize univfd_model:
    model = UnivFD("openai/clip-vit-large-patch14")
    
    # Initialize the optimizer:
    train_parameters = filter(lambda p: p.requires_grad, model.parameters())
    accelerator.print(f"trainable parameters: {train_parameters}, num: {len(train_parameters)}")
    optimizer = torch.optim.AdamW(
        train_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    
    # Initialize the dataloader:
    if args.dataset_name == "genimage":
        from dataset.genimage import Genimage
        image_dir = "/data_center/data2/dataset/chenwy/21164-data/genimage"
        generator_list = [ "ADM", "BigGAN", "Glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong" ]
        train_dataset = Genimage(accelerator, image_dir, "train", generator_list, train_real_transform, train_fake_transform)
        val_dataset = Genimage(accelerator, image_dir, "val", generator_list, val_real_transform, val_fake_transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=8, num_workers=8)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=8, num_workers=8)
    
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
    
    # Initialize the trackers:
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    init_kwargs = { "swanlab" : {"name": args.tracker_experiment_name} }
    tracker_config = dict(vars(args))
    accelerator.init_trackers(project_name=args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    BCELoss = torch.nn.BCEWithLogitsLoss()
    best_metric = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_dataloader, 
                            desc=f"Epoch {epoch+1}/{args.epochs} Training", 
                            disable=not accelerator.is_main_process)
        for uids, generators, image_path, images, labels in progress_bar:
            labels = labels.float().unsqueeze(1) # [batch_size, 1]
            optimizer.zero_grad()
            logits = model(images) # [batch_size, 1]
            loss = BCELoss(logits, labels)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            total_train_loss += loss.detach().float()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_train_loss = accelerator.gather(total_train_loss).mean() / len(train_dataloader)
        
        model.eval()
        correct_samples, total_samples = 0.0, 0.0
        with torch.no_grad():
            for uids, generators, image_path, images, labels in tqdm(val_dataloader, 
                                                                     desc=f"Val {epoch} / {len(val_dataloader)}", 
                                                                     disable=not accelerator.is_main_process):
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).long()
                gathered_preds = accelerator.gather(preds.squeeze())
                gathered_labels = accelerator.gather(labels.long())
                
                correct_samples += (gathered_preds == gathered_labels).sum().item()
                total_samples += gathered_labels.numel()

                
        avg_acc = correct_samples / total_samples
        
        accelerator.print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {avg_acc:.4f}")
        accelerator.log({
            "train_loss": avg_train_loss,
            "val_accuracy": avg_acc
        }, step=epoch)

        if best_metric < avg_acc and accelerator.is_main_process:
            best_metric = avg_acc
            accelerator.print(f"New best model found! Accuracy: {best_metric:.4f}")
            
            # Save the complete state of the training
            save_path = os.path.join(args.output_dir, "best_model")
            accelerator.save_state(save_path)
            accelerator.print(f"Saved model checkpoint to {save_path}")

    accelerator.end_training()