#!/bin/bash

INPUT_DIR="/root/autodl-tmp/Fast3Dcache/assets/sup"
OUTPUT_DIR="/root/autodl-tmp/Fast3Dcache/fast3Dcache/outputs"

if [ ! -d "$INPUT_DIR" ]; then
    echo "错误：输入目录 '$INPUT_DIR' 不存在。"
    echo "请先运行上一步的背景去除脚本。"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "在 '$INPUT_DIR' 中查找所有 .png 图片并开始处理..."
find "$INPUT_DIR" -type f -name "*.png" | while read image_path; do
    # a. 获取相对于 INPUT_DIR 的路径，结果如: airplane/airplane_000/0001.png
    relative_path="${image_path#$INPUT_DIR/}"
    
    # b. 移除 .png 后缀，结果如: airplane/airplane_000/0001
    relative_path_no_ext="${relative_path%.png}"
    
    # c. 将路径中的斜杠 / 替换为下划线 _，创建唯一的基础名，结果如: airplane_airplane_000_0001
    unique_base_name=$(echo "$relative_path_no_ext" | tr '/' '_')
    
    # d. 添加你的后缀，最终输出名如: airplane_airplane_000_0001_trellis_fast
    output_name="${unique_base_name}_demo_trellis"

    echo "-----------------------------------------------------"
    echo "正在处理图片: $image_path"
    echo "输出目录:     $OUTPUT_DIR"
    echo "输出基础名:   $output_name"
    echo "-----------------------------------------------------"
    
    # python -m example_f3c \
    #     --image_path "$image_path" \
    #     --output_dir "$OUTPUT_DIR" \
    #     --output_name "$output_name"

    python -m example_f3c --euler_steps 25 \
                      --use_f3c \
                      --image_path "$image_path" \
                      --output_name "$output_name" \
                      --output_dir "$OUTPUT_DIR" \
                      --full_sampling_ratio 0.2 \
                      --full_sampling_end_ratio 0.75 \
                      --assumed_slope -0.07

done