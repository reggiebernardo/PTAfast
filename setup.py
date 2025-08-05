from setuptools import setup

setup(
    name='PTAfast',
    version='0.0.2',
    description='Fast code for calculation of SGWB spatial correlations in PTA',
    url='https://github.com/reggiebernardo/PTAfast',
    author='Reginald Christian Bernardo',
    author_email='reginaldchristianbernardo@gmail.com',
    license='MIT',
    packages=['PTAfast'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'healpy',
        'astropy',
        'tqdm',
        'multiprocess',
        'py3nj'
    ],
    python_requires='>=3.10',
)

# setup(
#     name='PTAfast',
#     version='0.0.1',    
#     description='Fast code for calculation of SGWB spatial correlations in PTA',
#     url='https://github.com/reggiebernardo/PTAfast',
#     author='Reginald Christian Bernardo',
#     author_email='reginaldchristianbernardo@gmail.com',
#     license='MIT license',
#     packages=['PTAfast'],
#     install_requires=['numpy',
# 		      'scipy',                     
#               'py3nj',
#                      ],
# )
