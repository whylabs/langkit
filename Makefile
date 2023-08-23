.PHONY: default dist install test load-test pre-commit clean release


dist:
	poetry build

install:
	poetry install -E all

clean:
	rm -rf ./dist/langkit*.whl
	rm -rf ./dist/langkit*.tar.gz

test:
	poetry run pytest langkit/tests -o log_level=INFO -o log_cli=true

load-test:
	poetry run pytest langkit/tests -o log_level=WARN -o log_cli=true --load

pre-commit:
	poetry run pre-commit run --all-files

default: dist

release: clean dist install load-test
