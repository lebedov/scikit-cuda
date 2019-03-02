PYTHON := `which python`
DESTDIR = /

NAME = scikit-cuda
VERSION = $(shell $(PYTHON) -c 'import setup; print setup.VERSION')

.PHONY: package build docs install test clean

package:
	$(PYTHON) setup.py sdist --formats=gztar bdist_wheel

upload: | package
		twine upload dist/*

build:
	$(PYTHON) setup.py build

docs:
	$(PYTHON) setup.py build_sphinx

install:
	$(PYTHON) setup.py install --root=$(DESTDIR)

test:
	$(PYTHON) setup.py test

clean:
	$(PYTHON) setup.py clean
