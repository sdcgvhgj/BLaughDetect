import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="agkphysics/AudioSet",
    allow_patterns="data/bal_train/*.parquet",
    
    repo_type="dataset",
    local_dir="./data/AudioSet-bal_train",  # 保存到本地
)