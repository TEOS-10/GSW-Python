#!/bin/sh

PYTHONS=("cp37-cp37m" "cp38-cp38" "cp39-cp39")

for PYTHON in ${PYTHONS[@]}; do
    /opt/python/${PYTHON}/bin/pip install --upgrade pip wheel setuptools setuptools_scm pep517 twine auditwheel
    /opt/python/${PYTHON}/bin/pip install numpy==1.18
    /opt/python/${PYTHON}/bin/python -m pep517.build --source --binary . --out-dir /github/workspace/wheelhouse/
done

for whl in /github/workspace/wheelhouse/gsw*.whl; do
    auditwheel repair $whl
done
