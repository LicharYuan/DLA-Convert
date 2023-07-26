# dla connverter

This repo used to convert ONNX to CUDLA in  Orin / Xavier, as it required [cudla](https://docs.nvidia.com/cuda/cudla-api/index.html) lib.

## Usage

1. install

    ```
    mkdir build && cmake .. & make -j4
    ```

2. convert

    ```
    ./build/dla_converter/dla_int8  ./tests/data/test.onnx  ./tests/data/test.calib
    ./build/dla_converter/dla_fp16  ./tests/data/test.onnx  
    ```

    - To convert int8, a calibration file is required. You can refer to [TensorRT-Samples](https://github.com/NVIDIA/TensorRT/tree/498dcb009fe4c2dedbe9c61044d3de4f3c04a41b/samples/python) for more details.
    


## Known Issues

1. After installation, if you encounter the error `[eglUtils.cpp::operator()::105] Error Code 2: Internal Error (Assertion (eglCreateStreamKHR) != nullptr failed.)`, it is likely due to missing `nvidia-l4t-3d-core`. You can fix this by installing it using the following command: `apt install nvidia-l4t-3d-core -y `

    For more information, please refer to the discussion: [Internal Error Assertion Failed (eglCreateStreamKHR)](https://forums.developer.nvidia.com/t/internal-error-assertion-failed-eglcreatestreamkhr-nullptr/68405)



## Some Engineering problems 

1. When trying to run 2 DLA + 1 GPU Model on the same Orin device, it was observed that the GPU model becomes slower.

    This behavior seems to be normal. For more discussions and insight: https://forums.developer.nvidia.com/t/run-pure-conv2d-node-on-dla-makes-gpu-get-slower/219770/8
    
