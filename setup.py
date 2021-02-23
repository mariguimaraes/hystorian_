import setuptools
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

#if os.environ.get('CI_COMMIT_TAG'):
#    version = os.environ['CI_COMMIT_TAG']
#else:
#    version = os.environ['CI_JOB_ID']

setuptools.setup(
    name="hystorian",
    version="0.0.5",
    author="Loïc Musy <loic.musy@unige.ch>, Ralph Bulanadi <ralph.bulanadi@unige.ch>",
    author_email="loic.musy@unige.ch",
    description="a generic materials science data analysis Python package built with processing traceability, reproducibility, and archival ability at its core.",
    long_description=(here / 'README.md').read_text(encoding='utf-8'),
    url='https://gitlab.unige.ch/paruch-group/hystorian',
    packages=['hystorian', 'hystorian.io', 'hystorian.process']
    license='CC-By 4.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
