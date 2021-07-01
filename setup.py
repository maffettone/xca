from os import path
from setuptools import setup, find_packages
import sys
import versioneer
from subprocess import check_output, CalledProcessError


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 7)
if sys.version_info < min_version:
    error = """
xca does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(
        *(sys.version_info[:2] + min_version)
    )
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line
        for line in requirements_file.read().splitlines()
        if not line.startswith("#")
    ]

try:
    num_gpus = len(
        check_output(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], shell=True)
        .decode()
        .strip()
        .split("\n")
    )
    tf = "tensorflow-gpu>2.1.0" if num_gpus > 1 else "tensorflow>2.1.0"
except CalledProcessError:
    tf = "tensorflow>2.1.0"

requirements.append(tf)

setup(
    name="xca",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Crystallography companion agent",
    long_description=readme,
    author="Phillip Maffettone",
    author_email="pmaffetto@bnl.gov",
    url="https://github.com/maffettone/xca",
    python_requires=">={}".format(".".join(str(n) for n in min_version)),
    packages=find_packages(exclude=["docs", "tests"]),
    entry_points={
        "console_scripts": [
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        "xca": [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            "xca/examples/arxiv200800283/cifs*/*.cif"
        ]
    },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
)
