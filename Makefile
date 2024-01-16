.PHONY: default release dist install clean test load-test pre-commit lint-fix format-fix fix

default: help

dist: ## Build the wheel
	poetry build

install: ## Install all dependencies and extras
	poetry install -E all

clean:  ## Clean the project and generated files
	rm -rf ./dist/

test:  ## Run the tests
	poetry run pytest tests -o log_level=INFO -o log_cli=true

load-test:
	poetry run pytest langkit/tests -o log_level=WARN -o log_cli=true --load

lint: ## Check for type issues with pyright
	@{ echo "Running pyright\n"; poetry run pyright; PYRIGHT_EXIT_CODE=$$?; } ; \
	{ echo "\nRunning ruff check\n"; poetry run ruff check; RUFF_EXIT_CODE=$$?; } ; \
	exit $$(($$PYRIGHT_EXIT_CODE + $$RUFF_EXIT_CODE))

lint-fix: ## Fix lint issues with ruff
	poetry run ruff check --fix

format: ## Check for formatting issues
	poetry run ruff format --check

format-fix: ## Fix formatting issues
	poetry run ruff format

fix: lint-fix format-fix ## Fix all linting and formatting issues

bump-patch: ## Bump the patch version (_._.X) everywhere it appears in the project
	@$(call i, Bumping the patch number)
	poetry run bumpversion patch --allow-dirty

bump-minor: ## Bump the minor version (_.X._) everywhere it appears in the project
	@$(call i, Bumping the minor number)
	poetry run bumpversion minor --allow-dirty

bump-major: ## Bump the major version (X._._) everywhere it appears in the project
	@$(call i, Bumping the major number)
	poetry run bumpversion major --allow-dirty

bump-release: ## Convert the version into a release variant (_._._) everywhere it appears in the project
	@$(call i, Removing the dev build suffix)
	poetry run bumpversion release --allow-dirty

bump-build: ## Bump the build number (_._._-____XX) everywhere it appears in the project
	@$(call i, Bumping the build number)
	poetry run bumpversion build --allow-dirty


help: ## Show this help message.
	@echo 'usage: make [target] ...'
	@echo
	@echo 'targets:'
	@egrep '^(.+)\:(.*) ##\ (.+)' ${MAKEFILE_LIST} | sed -s 's/:\(.*\)##/: ##/' | column -t -c 2 -s ':#'

