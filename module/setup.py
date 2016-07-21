from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

extension_names = ["c_network", "c_network_trainer"]

for extension_name in extension_names:

   ext = [Extension(extension_name, sources=["{}.pyx".format(extension_name)], include_dirs=[np.get_include()])]
   setup(
      name=extension_name,
      cmdclass={'build_ext': build_ext},
      include_dirs=[np.get_include()],
      ext_modules=ext
      )