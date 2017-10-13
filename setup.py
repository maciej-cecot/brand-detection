from distutils.core import setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='wi-bd-api',
    version='0.1',
    install_requires=install_requires,
)

