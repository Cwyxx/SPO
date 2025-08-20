import os
import pandas as pd
from tqdm import tqdm
base_parquet_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/meta_data/pick_a_pic_v1/data"
base_pick_a_pic_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1"

def to_bytes(x):
    if x is None:
        return None
    if isinstance(x, bytes):
        return x
    if isinstance(x, (memoryview, bytearray)):
        return bytes(x)
    try:
        y = x.as_py()
        if isinstance(y, (bytes, bytearray, memoryview)):
            return bytes(y)
    except Exception:
        print(f"Error Processing bytes to image.")
    return None

def decide_winner(row):
    # skip row["are_different"] = True; row["are_different"] = False
    if pd.isna(row["are_different"]) or not bool(row["are_different"]):
        return None
    
    # check label
    l0, l1 = row.get("label_0"), row.get("label_1")
    if l0 not in (0, 1) or l1 not in (0, 1):
        return None
    
    if "best_image_id" in row and pd.notna(row["best_image_id"]):
        best_id = str(row["best_image_id"])
        uid0 = str(row["image_0_uid"]) if "image_0_uid" in row and pd.notna(row["image_0_uid"]) else None
        uid1 = str(row["image_1_uid"]) if "image_1_uid" in row and pd.notna(row["image_1_uid"]) else None
        if uid0 and best_id == uid0 and l0 == 1 and l1 == 0:
            return 0
        if uid1 and best_id == uid1 and l0 == 0 and l1 == 1:
            return 1

    return None

parquet_file_list = os.listdir(base_parquet_dir)
idx = 0
for parquet_idx, partquet_file in tqdm(enumerate(parquet_file_list), len=len(parquet_file_list), desc="parquet"):
    df = pd.read_parquet(os.path.join(base_parquet_dir, partquet_file))
    
    for row_idx, row in df.iterrows():
        winer = decide_winner(row)
        if winer is None:
            continue
        
        b0 = to_bytes(row.get("jpg_0"))
        b1 = to_bytes(row.get("jpg_1"))
        
        if b0 is None or b1 is None:
            continue