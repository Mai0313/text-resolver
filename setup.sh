#!/bin/bash

# 確認conda是否已經安裝
if ! command -v conda &> /dev/null
then
    echo "conda 沒有被安裝或找不到conda命令。請確認conda已經正確安裝並已添加到系統路徑。"
    exit
fi

# 創建名為torch2的conda環境，並設定python版本為3.9.15
conda create --name torch2 python=3.9.15 -y

# 啟動新的環境
source activate torch2

# 安裝pytorch和相關套件
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他需要的套件
pip install ipykernel ipywidgets pandas numpy autoroot hydra-core pytorch-lightning matplotlib opencv-python notebook pillow dvc-gdrive dvc
