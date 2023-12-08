.PHONY: default dist install test load-test pre-commit clean release


dist:
	poetry build

install:
	poetry install -E all

clean:
	rm -rf ./dist/langkit*.whl
	rm -rf ./dist/langkit*.tar.gz

test:
	# TODO why are the tests in langkit/tests?
	poetry run pytest langkit/tests -o log_level=INFO -o log_cli=true
	poetry run pytest tests -o log_level=INFO -o log_cli=true

load-test:
	poetry run pytest langkit/tests -o log_level=WARN -o log_cli=true --load

pre-commit:
	poetry run pre-commit run --all-files
	poetry run pyright


lint-fix:
	poetry run ruff check --fix

format-fix:
	poetry run ruff format

fix: lint-fix format-fix

default: dist

release: clean dist install load-test
