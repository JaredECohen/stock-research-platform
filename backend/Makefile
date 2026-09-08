# backend/Makefile — dependency lock maintenance.
#
# pyproject.toml [project.dependencies] is the only place a dependency is
# declared. requirements.txt is generated from it here and is what the
# Dockerfile, CI, nightly-live and the README install (`pip install -r
# requirements.txt`, unchanged). Never hand-edit requirements.txt: bump the
# floor in pyproject.toml, run `make lock`, commit both files.
#
# The lock is "universal": uv emits environment markers so the same file
# installs on the Docker image (python:3.12-slim, linux) and on a dev Mac.
# `--python-version 3.12` is the image's interpreter and the lock's floor;
# the package floor of 3.11 is only guaranteed to *parse* (see the
# py311-floor CI job), not to install from this lock.
#
# uv reads the existing requirements.txt as preferences, so re-running
# `make lock` with an unchanged pyproject.toml is a no-op: only a changed
# floor moves a pin. That property is what lets CI's lock-check diff the
# regenerated file against the committed one.

UV ?= uv
LOCK_PYTHON ?= 3.12

.PHONY: lock lock-check

lock:
	$(UV) pip compile pyproject.toml --universal --python-version $(LOCK_PYTHON) --no-header -o requirements.txt
	@{ \
	  printf '%s\n' \
	    '# GENERATED FILE — do not edit by hand.' \
	    '# Source of truth: pyproject.toml [project.dependencies]. Edit the floor' \
	    '# there, then run `make lock` from backend/ and commit both files.' \
	    '# CI (lock-check) fails when this file is stale.' \
	    '#' \
	    "# Produced by: uv pip compile pyproject.toml --universal --python-version $(LOCK_PYTHON)"; \
	  cat requirements.txt; \
	} > requirements.txt.tmp && mv requirements.txt.tmp requirements.txt

# Regenerates the lock and fails if that changed anything, i.e. someone
# edited pyproject.toml (or requirements.txt by hand) without re-locking.
lock-check: lock
	git diff --exit-code -- requirements.txt
