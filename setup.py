from setuptools import setup, find_packages

setup(
    name='image_merging',
    version='1.61',
    packages=find_packages(include=['models', 'models.*']),
    # install_requires=[
    #     'diffusers',
    #     'transformers',
    #     'accelerate',
    #     'xformers',
    #     # 'git+https://github.com/ChaoningZhang/MobileSAM.git'
    # ],
)