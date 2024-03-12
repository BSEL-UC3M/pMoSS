from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
# parse_requirements() returns generator of pip.req.InstallRequirement objects

setup(
    name='pmoss',
    packages=find_packages(include=['pmoss', 'pmoss.*']),
    version='2.1',
    license='BSD 3-Clause License',
    description='Python package to model the p-value as an n-dependent function using Monte Carlo cross-validation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="E. Gomez-de-Mariscal, V. Guerrero, A. Sneider, H. Hayatilaka, J.M. Phillip, D. Wirtz, A. Munoz-Barrutia",
    author_email='egomez@igc.gulbenkian.pt, mamunozb@ing.uc3m.es',
    url='https://github.com/BIIG-UC3M/pMoSS',
    download_url='https://github.com/BIIG-UC3M/pMoSS/archive/refs/tags/v2.1.tar.gz',
    keywords=['p-value', 'monte-carlo', 'statistical significance', 'null hypothesis testing', 'statistical test'],
    python_requires='>=3.9',
    install_requires=[
        'numpy==1.25.2',
        'pandas==1.5.3',
        'scipy==1.12.0',
        'xlrd==2.0.1',
        'matplotlib==3.8.3',
        'seaborn==0.13.2',
        'statsmodels==0.14.1',
        'glob2==0.7',
        'pytest-shutil==1.7.0',
        'openpyxl==3.1.2'
      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
)
