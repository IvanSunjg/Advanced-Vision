import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='avgmentations',
    version='0.0.6',
    author='Austin Pan, Gabriel Izsak, Ivan Sun, Zhonghao Wen',
    author_email='austinpan8@gmail.com',
    description='Data Augmentations for the Image Classification on Limited Data Challenge.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/IvanSunjg/Advanced-Vision/avgmentations',
    license='MIT',
    packages=['avgmentations'],
    install_requires=['numpy', 'matplotlib'],
)
