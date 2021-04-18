from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '1.1'

here = path.abspath(path.dirname(__file__))
OFFICIAL_URL = "https://github.com/penguinmenac3/leanai"
OFFICIAL_DOC_URL = f"{OFFICIAL_URL}/blob/main/"

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    parts = long_description.split("](")
    for idx in range(1, len(parts)):
        parts[idx] = OFFICIAL_DOC_URL + parts[idx]
    long_description = "](".join(parts)

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='leanai',
    version=__version__,
    description='A library to help writing ai functions with ease.',
    long_description=long_description,
    url=OFFICIAL_URL,
    download_url=OFFICIAL_URL + '/tarball/' + __version__,
    license='MIT',
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
    ],
    keywords='',
    packages=find_packages(exclude=['examples', 'docs', 'tests*']),
    include_package_data=True,
    author='Michael Fuerst',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='mail@michaelfuerst.de',
    entry_points={
        'console_scripts': [
            'leanai = leanai.core.cli_tools:main',
            'leanai_remote = leanai.core.cli_remote:main',
        ]
    },
    extras_require={
        'visualization': ['open3d==0.9.0.0', 'tensorboard'],
        'dev': ['nose2', 'packaging'],
        'all': ['open3d==0.9.0.0', 'tensorboard', 'nose2', 'packaging'],
    },
)
