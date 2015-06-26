from distutils.core import setup
from distutils.extension import Extension
import os
import sys
from os.path import join, exists,dirname
from os import environ
import pathlib
from subprocess import check_output

from Cython.Distutils import build_ext
have_cython = True
cmdclass = {'build_ext': build_ext}
libraries = [ 'ffts']

include_dirs = list(map(pathlib.Path.as_posix,pathlib.Path('./include').absolute().glob('**/')))
for lib in libraries:
    include_dirs.extend( map(lambda x: x.strip()[2:],check_output(['pkg-config','--cflags-only-I',lib]).split()))

suffix = '.a'
prefix = 'lib'

ff_extra_objects = ['ffts']
extra_objects = [join(check_output(['pkg-config','--variable=libdir',obj]).strip(),prefix+obj+suffix) for obj in ff_extra_objects]

mods = ['ffts']
extra_compile_args = ["-O4"]+check_output(["pkg-config","--cflags","--libs"]+libraries).split()

print(include_dirs)

ext_modules = [Extension( src_file.as_posix()[:-4].replace('/','.'),
    sources=[ src_file.as_posix()],
    include_dirs=include_dirs, extra_objects=extra_objects,
    extra_compile_args=extra_compile_args) for src_file in pathlib.Path('.').glob('*.pyx')]

for e in ext_modules:
    e.cython_directives = {"embedsignature": True,"boundscheck":False,"overflowcheck":False, "wraparound":True }

setup(name='ffts',
      version='0.0.1',
      author="gabriel d. karpman",
      license='MIT',
      description='A cython wrapper for the fastest fourier transform in the south.',
      classifiers=['Programming Language :: Python :: 2.7',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: BSD :: FreeBSD',
                   'Operating System :: POSIX :: Linux',
                   'Intended Audience :: Developers'],
      cmdclass=cmdclass, ext_modules=ext_modules)
