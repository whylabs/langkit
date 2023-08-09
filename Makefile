.PHONY: install test load-test pre-commit


install:
	poetry install -E all
	poetry run pip3 install torch torchvision torchaudio

test:
	poetry run pytest langkit/tests

load-test:
	poetry run pytest langkit/tests --load

pre-commit:
	poetry run pre-commit run --all-files
