from setuptools import setup, find_packages
import mlc

requirements = [
    'jupyter',
    'numpy',
    'matplotlib',
    'requests',
    'pandas',
    'sphinx',
    'torch',
    'torchvision',
    'xgboost==1.5',
    'tabulate',
]

setup(
    name='mlc',
    version=mlc.__version__,
    python_requires='>=3.7',
    author='MLC Developers',
    url='https://mlc.ai',
    description='Machine Learning Compiler',
    # license='MIT-0',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
