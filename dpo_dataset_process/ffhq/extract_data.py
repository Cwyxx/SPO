import os
import pandas as pd
import random
import json
from tqdm import tqdm
from PIL import Image

base_parquet_dir = "/data_center/data2/dataset/chenwy/21164-data/ffhq/meta_data"
sampled_images_dir = "/data_center/data2/dataset/chenwy/21164-data/ffhq/ffhq-10k"
caption_json_file_path = "/data_center/data2/dataset/chenwy/21164-data/ffhq/ffhq-10k-captions.json"
total_samples = 10000
random.seed(42)

os.makedirs(sampled_images_dir, exist_ok=True)

print("Scanning for Parquet files...")
all_parquet_files = [f for f in os.listdir(base_parquet_dir) if f.endswith('.parquet')]
if not all_parquet_files:
    raise ValueError(f"No Parquet files found in {base_parquet_dir}")

print("Loading all metadata into a single DataFrame...")
all_metadata_dfs = []
for parquet_file in tqdm(all_parquet_files, desc="Reading files"):
    parquet_file_path = os.path.join(base_parquet_dir, parquet_file)
    df = pd.read_parquet(parquet_file_path)
    all_metadata_dfs.append(df[['image', 'text']])

full_df = pd.concat(all_metadata_dfs, ignore_index=True)
print(f"Total records found: {len(full_df)}")


if len(full_df) < total_samples:
    print(f"Warning: Total records ({len(full_df)}) is less than requested samples ({total_samples}). Sampling all available records.")
    total_samples = len(full_df)

print(f"Sampling {total_samples} records...")
sampled_df = full_df.sample(n=total_samples, random_state=42)

captions = []
print("Copying images and generating captions file...")
for idx, row in tqdm(enumerate(sampled_df.itertuples(index=False), 1), total=total_samples, desc="Processing samples"):
    image_dict = row.image
    caption_text = row.text

    image_filename = f"{idx:05d}.png"
    destination_image_path = os.path.join(sampled_images_dir, image_filename)
    
    try:
        if isinstance(image_dict, dict) and 'bytes' in image_dict and image_dict['bytes'] is not None:
            image_bytes = image_dict['bytes']
            with open(destination_image_path, 'wb') as f:
                f.write(image_bytes)
            
            captions.append({
                "evalset_idx": idx,
                "caption": caption_text,
            })
        else:
            print(f"\nWarning: Skipping record {idx} due to invalid or empty image data. Data found: {image_dict}")
            
    except Exception as e:
        print(f"\nAn error occurred while processing and saving image {idx}: {e}")


print(f"Saving captions to {caption_json_file_path}...")
with open(caption_json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(captions, json_file, indent=4, ensure_ascii=False)

print("Done!")