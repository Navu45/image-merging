from setuptools import setup, find_packages

setup(
    name='image_merging',
    version='0.1',
    packages=find_packages('models/*'),
    install_requires=[
        'diffusers',
        'transformers',
        'accelerate'
    ],
)