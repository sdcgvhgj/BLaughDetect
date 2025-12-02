import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
os.environ['http_proxy'] = 'http://127.0.0.1:10809'
os.environ['https_proxy'] = 'http://127.0.0.1:10809'

from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="agkphysics/AudioSet",
    allow_patterns="data/bal_train/*.parquet",
    
    repo_type="dataset",
    local_dir="./data/AudioSet-bal_train",  # 保存到本地
)