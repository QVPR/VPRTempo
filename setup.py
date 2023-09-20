import os, sys

from setuptools import setup, find_packages


# define the base requires needed for the repo
requirements = [ 
                'matplotlib',
                'torch',
                'torchvision',
                'torchaudio',
                ]

# workaround as opencv-python does not show up in "pip list" within a conda environment
# we do not care as conda recipe has py-opencv requirement anyhow
is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
if not is_conda:
    requirements.append('opencv-python')

# define the setup
setup(
    name="VPRTempo-GPU",
    version="1.0.0",
    description='VPRTempo: A Fast Temporally Encoded Spiking Neural Network for Visual Place Recognition',
    author='Adam D Hines, Peter G Stratton, Michael Milford and Tobias Fischer',
    author_email='adam.hines@qut.edu.au',
    url='https://github.com/QVPR/VPRTempo',
    license='MIT',
    install_requires=requirements,
    python_requires='>=3.8',
    packages=find_packages(),
    keywords=['python', 'place recognition', 'spiking neural networks',
              'computer vision', 'robotics'],
    scripts=['VPRTempo.py'],
    package_data={'':['nordland_imageNames.txt','orc_imageNames.txt']}
)
