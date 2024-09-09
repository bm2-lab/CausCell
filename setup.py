from setuptools import setup, find_packages
from os import path
from io import open

# get __version__ from _version.py
ver_file = path.join('causcell', 'version.py')
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))

# read the contents of README.md
def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='causcell',
    version=__version__,
    description='CausCell for causal disentanglement of single-cell data',
    long_description=readme(), 
    long_description_content_type='text/markdown',
    url='https://github.com/bm2-lab/CausCell',
    author='Yicheng Gao, Kejing Dong, Caihua Shan, Dongsheng Li, and Qi Liu',
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    license='GPL-3.0 license'
)