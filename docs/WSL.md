# BitBLAS on WSL2

本指南记录了在 Windows 11 + WSL2 (Ubuntu 22.04) 上从源码构建 BitBLAS 以及使用当前仓库中已编译环境的方法。流程已经在 `CUDA 11.5 + conda 3.10` 的 WSL 来宾系统中验证。

## 1. 前置条件

- Windows 11 + WSL2（Ubuntu 22.04）  
- 最新 NVIDIA Windows 驱动 + WSL CUDA 工具链（`/usr/local/cuda` 指向 11.5+）  
- Conda（示例使用 `~/miniconda3`）  
- 已安装 `git`, `cmake`, `build-essential`

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config
```

## 2. 克隆仓库并拉取子模块

```bash
git clone https://github.com/microsoft/BitBLAS.git BitBLAS-Quant
cd BitBLAS-Quant
git submodule update --init --recursive
```

## 3. 准备 Conda 环境 + LLVM

推荐创建独立环境（或复用现有 `gr00t` 环境）：

```bash
conda create -n bitblas python=3.10 -y
conda activate bitblas

# 基础依赖
pip install -r requirements.txt

# LLVM 17，提供 llvm-config 与共享库
conda install -y llvmdev=17.0.6
```

### 目录变量

文档后续假设：

```bash
export BITBLAS_ROOT=/mnt/c/Users/ASUS/Documents/GitHub/BitBLAS-Quant
export CONDA_ENV=bitblas      # 若使用自定义环境名请自行替换
export LLVM_CONFIG=$(conda run -n $CONDA_ENV which llvm-config)
```

## 4. 构建 TVM

```bash
cd $BITBLAS_ROOT/3rdparty/tvm
mkdir -p build
cp cmake/config.cmake build
cd build

cat <<EOF >> config.cmake
set(USE_LLVM $LLVM_CONFIG)
set(USE_CUDA /usr/local/cuda)
EOF

cmake ..
cmake --build . -j$(nproc)
```

> **提示**：如果之前构建失败，建议 `rm -rf build` 后再执行上述步骤。

## 5. 构建 TileLang

```bash
cd $BITBLAS_ROOT/3rdparty/tilelang
cmake -S . -B build -DTVM_PREBUILD_PATH=$BITBLAS_ROOT/3rdparty/tvm/build
cmake --build build -j$(nproc)
```

## 6. 配置环境变量

在当前终端（或写入 `~/.bashrc`）：

```bash
export PYTHONPATH=$BITBLAS_ROOT/3rdparty/tvm/python:$BITBLAS_ROOT/3rdparty/tilelang:$BITBLAS_ROOT:$PYTHONPATH
```

如果需要，也可把以下命令加入 `~/.bashrc` 中：

```bash
export BITBLAS_ROOT=/mnt/c/Users/ASUS/Documents/GitHub/BitBLAS-Quant
export PYTHONPATH=$BITBLAS_ROOT/3rdparty/tvm/python:$BITBLAS_ROOT/3rdparty/tilelang:$BITBLAS_ROOT:$PYTHONPATH
```

## 7. 验证

```bash
cd $BITBLAS_ROOT
python -c "import bitblas; print(bitblas.__version__)"
```

首次运行时 TileLang 会自动编译 Cython JIT 适配层，日志中出现 `Compiling cython jit adapter` 属正常现象。

## 8. 复用本机现成环境

当前仓库已经在 `~/miniconda3/envs/gr00t` 内完成 TVM/TileLang 构建，可直接按以下步骤启动：

```bash
cd /mnt/c/Users/ASUS/Documents/GitHub/BitBLAS-Quant
conda activate gr00t
export PYTHONPATH=$PWD/3rdparty/tvm/python:$PWD/3rdparty/tilelang:$PWD:$PYTHONPATH

python -c "import bitblas; print(bitblas.__version__)"
```

如需在 VSCode/脚本中使用，只需激活 `gr00t` 环境并保持上述 `PYTHONPATH` 设置即可。
