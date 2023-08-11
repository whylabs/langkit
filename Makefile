.PHONY: install test load-test pre-commit


install:
	poetry install -E all

test:
	poetry run pytest langkit/tests

load-test:
	poetry run pytest langkit/tests -o log_level=INFO -o log_cli=true --load

pre-commit:
	poetry run pre-commit run --all-files
