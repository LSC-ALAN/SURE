# SURE_ONNX_CPP

## Usage:

use CPU (default)
```
wget -c https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -zxvf onnxruntime-linux-x64-1.18.0.tgz -C third_party/
rm onnxruntime-linux-x64-1.18.0.tgz
```

or use GPU
```
wget -c https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-1.18.0.tgz
tar -zxvf onnxruntime-linux-x64-gpu-1.18.0.tgz -C third_party/
rm onnxruntime-linux-x64-gpu-1.18.0.tgz
```
then modify CMakeLists.txt lines 9-12, and Uncomment the lines 15-20 in sure/sure.cpp


run demo 
```
mkdir build && cd build
cmake ..
make -j8
cd ..
sh demo.sh

```
