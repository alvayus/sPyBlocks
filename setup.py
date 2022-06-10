from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sPyBlocks',
    version='0.0.1',
    author="Alvaro Ayuso-Martinez",
    author_email="aayuso@us.es",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alvayus/spyblocks",
    packages=find_packages(),
    install_requires=[
        'sPyNNaker>=1!6.0.0', 'numpy>=1.22.1', 'matplotlib>=3.5.1', 'XlsxWriter>=3.0.2'
    ],
)
