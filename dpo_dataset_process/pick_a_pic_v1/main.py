import os
import pandas as pd
import json
from tqdm import tqdm
base_parquet_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/meta_data/pick_a_pic_v1/data"
base_pick_a_pic_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1"
id_start_prefix = "train-00000"

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
    
    # check label (not 0.5)
    try:
        l0 = float(row.get("label_0"))
        l1 = float(row.get("label_1"))
    except Exception as e:
        return None
    
    if l0 not in (0.0, 1.0) or l1 not in (0.0, 1.0):
        return None
    l0, l1 = int(l0), int(l1)

    if "best_image_uid" in row and pd.notna(row["best_image_uid"]):
        best_uid = str(row["best_image_uid"])
        uid0 = str(row["image_0_uid"]) if "image_0_uid" in row and pd.notna(row["image_0_uid"]) else None
        uid1 = str(row["image_1_uid"]) if "image_1_uid" in row and pd.notna(row["image_1_uid"]) else None
        if uid0 and best_uid == uid0 and l0 == 1 and l1 == 0:
            return 0
        if uid1 and best_uid == uid1 and l0 == 0 and l1 == 1:
            return 1

    return None

parquet_file_list = [ f for f in os.listdir(base_parquet_dir) if f.startswith("train") ]
for parquet_idx, partquet_file in tqdm(enumerate(parquet_file_list), position=0, dynamic_ncols=True, total=len(parquet_file_list), desc="parquet"):
    df = pd.read_parquet(os.path.join(base_parquet_dir, partquet_file))
    
    for row_idx, row in tqdm(df.iterrows(), position=1, dynamic_ncols=True, total=len(df), desc=partquet_file):
        # check winer.
        winner_idx = decide_winner(row)
        if winner_idx is None:
            continue
        
        # check image.
        b0 = to_bytes(row.get("jpg_0"))
        b1 = to_bytes(row.get("jpg_1"))
        if b0 is None or b1 is None:
            continue
        
        # get image id
        uid0 = row.get("image_0_uid")
        uid1 = row.get("image_1_uid")
        if pd.isna(uid0) or pd.isna(uid1):
            continue
        uid0 = str(uid0)
        uid1 = str(uid1)
        
        # ge prompt:
        prompt = row.get("caption")
        if pd.isna(prompt):
            continue
        prompt = str(prompt)
        
        write_id = f"{partquet_file[0:len(id_start_prefix)]}-row_{row_idx}"
        write_dir = os.path.join(base_pick_a_pic_dir, write_id)
        os.makedirs(write_dir, exist_ok=True)
        
        if winner_idx == 0:
            win_image_id, lose_image_id = uid0, uid1
            win_bytes, lose_bytes = b0, b1
        else:
            win_image_id, lose_image_id = uid1, uid0
            win_bytes, lose_bytes = b1, b0

        win_filename = f"{win_image_id}.png"
        lose_filename = f"{lose_image_id}.png"

        with open(os.path.join(write_dir, win_filename), "wb") as f:
            f.write(win_bytes)
        with open(os.path.join(write_dir, lose_filename), "wb") as f:
            f.write(lose_bytes)

        # 写 info.json
        info = {
            "id": write_id,
            "prompt": prompt,
            "win_image_id": win_image_id,
            "lose_image_id": lose_image_id,
        }
        with open(os.path.join(write_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)