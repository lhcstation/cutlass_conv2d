## Installation

### Install Cutlass


```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass 
mkdir build & cd build
cmake .. -DCUTLASS_NVCC_ARCHS=89 -DCMAKE_BUILD_TYPE=Release
make -j
```

- `-DCUTLASS_NVCC_ARCHS=80`: 指定目标架构（如你的 NVIDIA 4090，应该使用 80 或 90）。
其他架构选项：75（Turing）、80（Ampere）、90（Ada Lovelace）

- `-DCMAKE_BUILD_TYPE=Release`: 以 Release 模式构建，优化性能

### File Structure

```bash
.
├── custom_conv.cu
├── cutlass
├── model.py
├── Readme.md
└── setup.py

1 directory, 4 files
```

### Install Custom_conv

```bash
python setup.py install 
```


## Test

```bash
python model.py
```
