import os
import pandas as pd

validation_parquet_path = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/meta_data/pick_a_pic_v1/data/validation_unique-00000-of-00001-c7e520171372df72.parquet"
df = pd.read_parquet(validation_parquet_path)


