#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

// Data types for input and output tensors
// and computation between elements
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;


#define CUTLASS_CHECK(status)                                                                               \
    {                                                                                                       \
        cutlass::Status error = status;                                                                     \
        if (error != cutlass::Status::kSuccess) {                                                           \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__      \
                    << std::endl;                                                                           \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
    }

// Whether to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// SM architecture number
using SmArch = cutlass::arch::Sm89;

// Threadblock tile shape
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;

// Warp tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;

// MMA (Tensor Core instruction, in this case) tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// How the kernel schedules threadblocks
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipeline stages to use
constexpr int NumStages = 3;

// Which iterator algorithm to use: Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// Is the output packed or strided
// Use kStride if using strided output
static cutlass::conv::StrideSupport const OutputStride = cutlass::conv::StrideSupport::kUnity;

// The epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// Kernel properties type
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm,
    OutputStride
>::Kernel;

// Type of the actual kernel
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

// CUDA 内核函数
torch::Tensor custom_conv_forward(
    torch::Tensor input, torch::Tensor filter,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h=1, int pad_w=1,
    int stride_h=1, int stride_w=1,
    int dilation_h=1, int dilation_w=1)
{   
    cutlass::Tensor4DCoord input_size(N, H, W, C);
    cutlass::Tensor4DCoord filter_size(K, R, S, C);
    cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
    cutlass::MatrixCoord conv_stride(stride_h, stride_w);
    cutlass::MatrixCoord dilation(dilation_h, dilation_w);
    ElementComputeEpilogue alpha(1);
    ElementComputeEpilogue beta(0);

    cutlass::Tensor4DCoord output_size = cutlass::Tensor4DCoord(
        input_size.n(),
        (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
        (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
        filter_size.n()
    );

    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(input_size);
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(filter_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(output_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(output_size);

    const void* input_data_ptr = input.data_ptr();
    size_t torch_data_size = input.numel() * sizeof(ElementInputA);

    // 获取 CUTLASS HostTensor 的设备指针
    void* tensor_a_data_ptr = tensor_a.device_data();
    size_t tensor_a_data_size =tensor_a.capacity() * sizeof(ElementInputA);

    // 使用 cudaMemcpy 将数据从 torch::Tensor 复制到 CUTLASS HostTensor
    cudaMemcpy(tensor_a_data_ptr, input_data_ptr, torch_data_size, cudaMemcpyDeviceToDevice);

    const void* filter_data_ptr = filter.data_ptr();
    size_t filter_data_size = filter.numel() * sizeof(ElementInputB);

    // 获取 CUTLASS HostTensor 的设备指针
    void* tensor_b_data_ptr = tensor_b.device_data();
    size_t tensor_b_data_size =tensor_b.capacity() * sizeof(ElementInputB);

    // 使用 cudaMemcpy 将数据从 torch::Tensor 复制到 CUTLASS HostTensor
    cudaMemcpy(tensor_b_data_ptr, filter_data_ptr, filter_data_size, cudaMemcpyDeviceToDevice);

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Construct Conv2dProblemSize with user defined output size
    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        mode,
        split_k_slices
    );

    // Construct ImplicitGemm::Argument structure with conv2d
    // problem size, data pointers, and epilogue values
    typename ImplicitGemm::Arguments arguments{
        problem_size,
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        tensor_c.device_ref(),
        tensor_d.device_ref(),
        {alpha, beta},
    };  
    ImplicitGemm implicit_gemm_op;
    cutlass::Status status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);


    //
    // Initialize CUTLASS Convolution
    //
    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    //
    // Launch initialized CUTLASS kernel
    //
    status = implicit_gemm_op();

    CUTLASS_CHECK(status);    

    auto shape = tensor_d.extent();
    int n = shape.n();  // Batch size
    int h = shape.h();  // Height
    int w = shape.w();  // Width
    int c = shape.c();  // Channels

    // 创建一个 PyTorch Tensor（NHWC 格式）
    torch::Tensor output = torch::empty({n, h, w, c}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 获取 CUTLASS HostTensor 的数据指针
    const void* cutlass_data_ptr = tensor_d.device_data();
    size_t cutlass_data_size = tensor_d.capacity() * sizeof(ElementOutput);

    // 获取 PyTorch Tensor 的数据指针
    void* torch_data_ptr = output.data_ptr();

    // 将 CUTLASS 数据复制到 PyTorch Tensor
    cudaMemcpy(torch_data_ptr, cutlass_data_ptr, cutlass_data_size, cudaMemcpyDeviceToDevice);

    return output;

}
