#!/bin/sh

# if testing locally try the docker image interactively
# docker run --net=host -it --rm -v $(pwd):/home/ quay.io/pypa/manylinux2010_x86_64

PYTHONS=("cp38-cp38" "cp39-cp39" "cp310-cp310")

for PYTHON in ${PYTHONS[@]}; do
    /opt/python/${PYTHON}/bin/pip install --upgrade pip wheel setuptools setuptools_scm pep517 twine auditwheel
    /opt/python/${PYTHON}/bin/pip install oldest-supported-numpy
    /opt/python/${PYTHON}/bin/python -m build --sdist --wheel . --outdir /github/workspace/wheelhouse/
done

for whl in /github/workspace/wheelhouse/gsw*.whl; do
    auditwheel repair $whl
done
