[Pytorch help](https://pytorch.org/get-started/locally/)

Server only:

```
cryptography~=43.0.1
```

Win nvidia:

```ps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib~=3.9.2 pillow~=10.2.0 face-recognition~=1.3.0
```

Linux AMD:

```sh
export HSA_OVERRIDE_GFX_VERSION=11.0.0 # fix for RX 7800 XT
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
pip3 install matplotlib~=3.9.2 pillow~=10.2.0 face-recognition~=1.3.0
```
