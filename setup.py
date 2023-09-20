from setuptools import setup, find_packages


# define the base requires needed for the repo
requirements = ['opencv-python', 
                'matplotlib',
                'torch',
                'torchvision',
                'torchaudio',
                ]

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
    scripts=['VPRTempo.py',
             'src/blitnet.py',
             'src/metrics.py',
             'src/nordland.py',
             'src/utils.py',
             'src/validation.py'],
    package_data={'':['nordland_imageNames.txt','orc_imageNames.txt']}
)