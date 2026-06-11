# SURE_DEPLOY

environment
```
pip install -r requirements_deploy.txt
```

export onnx model from torch .pth according to your needs (Height Width TopK)

```
python export_onnx.py
```


run demo
```
run_onnx.py
```


And C++ inference demo: sure_onnx_cpp