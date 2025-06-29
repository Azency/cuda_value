# setup.py ─ 只依赖 setuptools + NVCC + GCC
import os, sysconfig, pathlib, numpy, sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")  # 可自行 export
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "/home/pengbo/miniconda3")
PYTHON_VERSION = sys.version.split(".")[1]

class CUDAExtension(Extension):
    """简单占位，不做任何编译"""
    def __init__(self, name, sources):
        super().__init__(name, sources=sources)
        self.include_dirs = [numpy.get_include(),
                             os.path.join(CUDA_HOME, "include"),
                             os.path.join(CONDA_PREFIX, f"include/python3.{PYTHON_VERSION}"),
                             "src"]                       # 可按需调整
        self.library_dirs = [os.path.join(CUDA_HOME, "lib64")]
        self.libraries    = ["cudart"]
        # self.marco_params_path = "src/config.txt"

class BuildExt(build_ext):
    def build_extension(self, ext):
        objects = []
        for src in ext.sources:
            if src.endswith(".cu"):
                objects.append(self.compile_cuda(src, ext))
            else:                               # .c / .cpp
                objects.append(self.compile_host(src, ext))
        self.link(ext, objects)

    # --------- 定义编译前定义宏参数 ---------
    def define_macro_params(self, ext):
        pass
    # -------- nvcc 编译 .cu → .o --------
    def compile_cuda(self, src, ext):
        obj_path = self.obj_path(src)
        nvcc = f"{CUDA_HOME}/bin/nvcc"
        cmd  = [nvcc,
                "-O3", "--use_fast_math",
                "-arch=sm_86",                # RTX-3090, 改成你实际 GPU
                "-std=c++17",                 # 主机端仍走 C++ 解析
                "-Xcompiler", "-fPIC",
                "-c", src, "-o", obj_path] + \
               sum([["-I", d] for d in ext.include_dirs], [])
        self.spawn(cmd)
        return obj_path

    # -------- gcc/clang 编译 .c → .o -----
    def compile_host(self, src, ext):
        obj_path = self.obj_path(src)
        cc  = sysconfig.get_config_var("CC").split()[0]
        cmd = [cc, "-O3", "-std=c11", "-fPIC",
               "-c", src, "-o", obj_path] + \
              sum([["-I", d] for d in ext.include_dirs], [])
        self.spawn(cmd)
        return obj_path

    # -------- 链接所有 .o → .so ----------
    # def link(self, ext, objects):
    #     cc  = sysconfig.get_config_var("CC").split()[0]
    #     ext_path = self.get_ext_fullpath(ext.name)
    #     cmd = [cc, "-shared", "-o", ext_path, *objects] + \
    #           sum([["-L", d] for d in ext.library_dirs], []) + \
    #           ["-l"+lib for lib in ext.libraries]
    #     self.spawn(cmd)
    def link(self, ext, objects):
        # 确保输出目录存在
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        
        cc = sysconfig.get_config_var("CC").split()[0]
        cmd = [cc, "-shared", "-o", ext_path, *objects] + \
            sum([["-L", d] for d in ext.library_dirs], []) + \
            ["-l"+lib for lib in ext.libraries]
        self.spawn(cmd)

    # -------- 生成中间 .o 路径 -----------
    def obj_path(self, src):
        build_tmp = pathlib.Path(self.build_temp)
        build_tmp.mkdir(parents=True, exist_ok=True)
        return str(build_tmp / (pathlib.Path(src).name + ".o"))

# ---------------------------------------------------------------------
setup(
    name="compute_l",
    version="0.1.0",
    ext_modules=[CUDAExtension("compute_l",
        ["src/wrapper.c",          # CPython 接口
         "src/pycomputel.cu",        # 你给出的主逻辑
         "src/cuda_value.cu" ])],  # 其它 kernel & 工具
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
