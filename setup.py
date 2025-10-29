from setuptools import setup, find_packages
from sand_ssl._version import __version__

setup(
    name = 'sand_ssl.py',
    packages = find_packages(),
    author = 'Daniela Wiepert',
    python_requires='>=3.8',
    install_requires=[
      "huggingface-hub==0.35.3",
      "librosa==0.11.0",
      "numpy",
      "openpyxl",
      "pandas",
      "scikit-learn",
      "scipy",
      "torch",
      "torchaudio",
      "torchvision",
      "tqdm",
      "transformers"
    ],
    include_package_data=False,
    version = __version__
)
