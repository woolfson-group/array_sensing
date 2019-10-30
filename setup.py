
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='array_sensing',
    packages=find_packages(),
    version='0.1.0',
    description=('BADASS_ML: A package to parse the input data and run ML '
                 'on fluorescence data collected from the BADASS (Barrel '
                 'Array Diagnostics And SenSing) technology'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=('Kathryn Shelley, Chris Wells Wood and Will Dawson in the lab '
            'of Professor Dek Woolfson, University of Bristol'),
    author_email='kathryn.l.shelley@gmail.com',
    url='https://github.com/woolfson-group/sensing_array_paper_2019',
    license='MIT',
    keywords=['badass', 'array sensing', 'machine learning'],
    install_requires=['jinja2', 'numpy', 'matplotlib', 'pandas', 'scikit-learn',
                      'scipy', 'seaborn', 'xlrd'],
    classifiers=['Programming Language :: Python'],
    python_requires=('!=2.*, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, '
                     '!=3.5.*, <4')
)
