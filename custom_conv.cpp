#include <torch/extension.h>

#include "cutlass/cutlass.h"


// 声明 CUDA 内核
torch::Tensor custom_conv_forward(
    torch::Tensor input, torch::Tensor weight,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h=1, int pad_w=1,
    int stride_h=1, int stride_w=1,
    int dilation_h=1, int dilation_w=1);

// PyTorch C++ 封装
torch::Tensor custom_conv_forward_launcher(
    torch::Tensor input, torch::Tensor filter,
    int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
    
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(3);
    int K = filter.size(0);
    int R = filter.size(1);
    int S = filter.size(2);

    return custom_conv_forward(
        input, filter,
        N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv_forward", &custom_conv_forward_launcher, "Custom Convolution Forward");
}
