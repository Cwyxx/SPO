import os
import pandas as pd
import json

caption_file_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1/pick_a_pic_validation_prompt_500.json"
validation_parquet_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/meta_data/pick_a_pic_v1/data/validation_unique-00000-of-00001-c7e520171372df72.parquet"
df = pd.read_parquet(validation_parquet_path)

caption_data = []
for idx, row in df.iterrows():
    caption_data.append({
        "evalset_idx": idx,
        "caption": row['caption']  # 假设 'caption' 是列名
    })

with open(caption_file_path, "w") as json_file:
    json.dump(caption_data, json_file, indent=4)

print(f"Data has been saved to {caption_file_path}")
print(f"Data num {len(caption_data)}")