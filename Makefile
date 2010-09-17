PYTHON = /usr/bin/python
DESTDIR = /usr

NAME = scikits.cuda
VERSION = $(shell $(PYTHON) -c 'import setup; print setup.VERSION')
LANG = python

.PHONY: package build install clean

package:
	$(PYTHON) setup.py sdist --formats=gztar && \
	mv -f dist/$(NAME)-$(VERSION).tar.gz dist/$(NAME)-$(LANG)-$(VERSION).tar.gz 

build:
	$(PYTHON) setup.py build

install:
	$(PYTHON) setup.py install --root=$(DESTDIR)

clean:
	$(PYTHON) setup.py clean
