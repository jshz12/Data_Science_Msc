from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(['/content/drive/MyDrive/Segunda Entrega NLP /skseq/sequences/structured_perceptron_c.pyx'])
)