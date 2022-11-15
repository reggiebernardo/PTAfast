from setuptools import setup

setup(
    name='PTAfast',
    version='0.0.0',    
    description='Fast code for calculation of SGWB spatial correlations in PTA',
    url='https://github.com/reggiebernardo/PTAfast',
    author='Reginald Christian Bernardo',
    author_email='reginaldchristianbernardo@gmail.com',
    license='MIT license',
    packages=['PTAfast'],
    install_requires=['numpy',
		      'scipy',                     
                      ],
)