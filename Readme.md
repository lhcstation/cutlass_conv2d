## Installation

### Install Cutlass


```bash
git clone https://github.com/NVIDIA/cutlass.git
```

### File Structure

```bash
.
├── custom_conv.cpp
├── custom_conv_kernel.cu
├── cutlass
├── model.py
├── Readme.md
└── setup.py

1 directory, 4 files
```

### Install Custom_conv

```bash
pip install -e .
```


## Test

```bash
python model.py
```