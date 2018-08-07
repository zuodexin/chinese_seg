set -e
while true; do
    read -p '使用现有的字向量?(否则会用自己的数据训练字向量)(Y/N):' yn
    case $yn in
        [Yy]* ) python deepmodel.py; break;;
        [Nn]* ) 
                bash train_embedding.sh
                python deepmodel.py -e
                break;;
        * ) echo "请回答 yes 或 no.";;
    esac
done

