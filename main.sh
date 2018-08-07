#!/bin/bash
set -e

echo 欢迎使用
echo 本程序可以使用两种不同的算法分词
echo 选择您希望使用的分词算法
echo 1 基于crf的算法
echo 2 基于字向量+BiLSTM+CRF的算法
while true; do
    read -p "请输入数字：" option
    case $option in
        1)  echo "1 训练crf"
            echo "2 测试crf"
            read -p "请输入数字：" option
            case $option in
                1)  bash train_crf.sh;;
                2)  bash test_crf.sh;;
                *)  echo "输入有误";;
            esac
            break;;
        2)  echo "1 训练深度模型"
            echo "2 测试深度模型"
            read -p "请输入数字：" option
            case $option in
                1)  bash train_deep.sh;;
                2)  bash test_deep.sh;;
                *)  echo "输入有误";;
            esac
            break;;
        *)  echo 输入有误，重新输入;;
    esac
done