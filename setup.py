from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import pybind11

class custom_build_ext(build_ext):
    def build_extensions(self):
        nvcc = 'nvcc'
        for ext in self.extensions:
            cuda_sources = [s for s in ext.sources if os.path.splitext(s)[1] == '.cu']
            other_sources = [s for s in ext.sources if os.path.splitext(s)[1] != '.cu']
            ext.sources = other_sources

            extra_objects = ext.extra_objects or []
            for source in cuda_sources:
                obj_file = os.path.splitext(source)[0] + '.o'
                print('Compiling CUDA source:', obj_file)
                include_dirs = ext.include_dirs or []
                include_args = [item for sublist in [['-I', inc] for inc in include_dirs] for item in sublist]

                nvcc_flags = [
                    '-c', source,
                    '-o', obj_file,
                    '-Xcompiler', '-fPIC',
                    '-std=c++17',
                    '-arch=sm_75',  # Adjust based on your GPU
                    '--compiler-options', '-fPIC'
                ] + include_args

                subprocess.check_call([nvcc] + nvcc_flags)
                extra_objects.append(obj_file)

            ext.extra_objects = extra_objects

        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'eigenvalues_cuda',
        sources=[
            'ext.cpp',
            'trignometric_soln.cu',
        ],
        include_dirs=[
            pybind11.get_include(),
            '/usr/local/cuda/include',
        ],
        library_dirs=[
            '/usr/local/cuda/lib64',
        ],
        libraries=['cudart'],
        extra_compile_args=['-std=c++17'],
        language='c++',
    ),
]

setup(
    name='eigenvalues_cuda',
    version='0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)