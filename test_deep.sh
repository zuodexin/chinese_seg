#!/bin/bash

set -e

echo 1 PLOYGLOT embedding
echo 2 自己训练的embedding
while true; do
    read -p "请输入数字：" option
    case $option in
        1)  polyglot=true
            break;;
        2)  polyglot=false
            break;;
        *) echo "输入有误，重新输入";;
    esac
done

echo 1 测试各项数值指标
echo 2 测试在输入文本上的表现
while true; do
    read -p "请输入数字：" option
    case $option in
        1)  
            if $polyglot; then
                python evaluate_deep.py -e output/data/PKU1998_01_test.tfrecords -p
            else
                python evaluate_deep.py -e output/data/PKU1998_01_test.tfrecords
            fi
            break;;
        2)  read -p "请输入待分词文件的路径：" filename
            if [ ! -f "$filename" ]; then
                echo 文件$filename不存在
                break
            fi
            if $polyglot; then
                python evaluate_deep.py -t $filename -p
            else
                python evaluate_deep.py -t $filename
            fi
            echo "结果保存在 output/deep_partitioned.txt"
            break;;
        *) echo "输入有误，重新输入";;
    esac
done