#!/bin/bash

# 训练原始模型
echo "=== 训练原始PWA模型 ==="
python script/run.py -c config/PWA.yaml --gpus [0]
# 训练原始模型
echo "=== 训练原始HGA模型 ==="
python script/run.py -c config/HGA.yaml --gpus [0]


