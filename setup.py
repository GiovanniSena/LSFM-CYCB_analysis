import os, io

from setuptools import setup, find_packages

setup(
    name='lightroot',
    version='0.9',
    author='Sirsh',
    author_email='amartey@gmail.com',
    license='MIT',
    url='git@gitlab.com/Amarteifio/lightroot',
    keywords='root microscopy tracking',
    description='lighroot description',
    long_description=('lightroot description'),
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points={
        'console_scripts': [
            'lightroot = lightroot.__main__:main'
            ],
    },
    classifiers=[
        'Development Status :: Beta',
        'Intended Audience :: Developers',
        'License :: MIT',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Communications :: Chat',
        'Topic :: Internet',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)


