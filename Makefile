# One command runs everything: `make all`
PYTHON ?= python

.PHONY: all test hooks clean

all:
	$(PYTHON) run_all.py

test:
	$(PYTHON) -m pytest -q

# Install the pre-commit secret scan into this clone.
hooks:
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

clean:
	rm -rf outputs
