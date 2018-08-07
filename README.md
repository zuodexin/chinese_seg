### 环境搭建
#### 安装CRF++
- 下载[CRF++-0.58.tar.gz](https://taku910.github.io/crfpp/#download)
- `tar zxvf CRF++-0.58.tar.gz`
- `cd CRF++-0.58`
- `./configure`
- `make` 
- `su`
- `make install`
- `cd python`
- `python setup.py install`

#### 安装TensorFlow
按照[tensorflow官网的指导](https://www.tensorflow.org/install/install_linux)进行安装


### 数据准备


```bash
bash ./download_dataset.sh
bash ./prepare_data.sh
```

### 主程序
```
bash main.sh
```

### 基于CRF的算法

#### crf训练

```bash
bash train_crf.sh
```
CRF model will be saved to `./output/models/PKU.model`

#### crf测试

```bash
bash test_crf.sh
```

### 基于神经网络的算法


#### 训练BiLSTMCRF
```bash
bash train_deep.sh
```

#### 测试深度模型
```bash
bash test_deep.sh
```