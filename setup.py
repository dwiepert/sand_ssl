from setuptools import setup, find_packages
from sand_ssl._version import __version__

setup(
    name = 'sand_ssl.py',
    packages = find_packages(),
    author = 'Daniela Wiepert',
    python_requires='>=3.10',
    install_requires=[
       "huggingface-hub==0.35.3",
      "librosa==0.11.0",
      "numpy==2.2.6",
      "openpyxl==3.1.5",
      "pandas==2.3.3",
      "scikit-learn==1.7.2",
      "scipy==1.15.3",
      "torch==2.8.0",
      "torchaudio==2.8.0",
      "torchvision==0.23.0",
      "tqdm==4.67.1",
      "transformers==4.57.0"
    ],
    include_package_data=False,
    version = __version__
)
