from setuptools import setup, find_packages

setup(
    name='spock-literature',  
    version='0.0.7',
    author='Youssef Briki',
    author_email='youssef.briki05@gmail.com',
    description='A description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AccelerationConsortium/spock.git',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
