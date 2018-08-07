#!/bin/bash

#根据CRF++的安装位置更改
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./CRF++-0.58/.lib
export PATH=$PATH:./CRF++-0.58

set -e
echo "1 测试各项数值指标"
echo "2 测试在输入文本上的表现"
while true; do
    read -p "请输入数字：" option
    case $option in
        1)  python evaluate.py output/model/PKU.model -e output/data/test.txt; 
            break;;
        2)  read -p "请输入待分词文件的路径：" filename
            if [ ! -f "$filename" ]; then
                echo 文件$filename不存在
                break
            fi
            python evaluate.py output/model/PKU.model -t $filename
            echo "结果保存在 output/result.txt"
            break;;
        *) echo "输入有误，重新输入";;
    esac
done
