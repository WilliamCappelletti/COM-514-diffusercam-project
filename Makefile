.PHONY: env requirements

env: 
	# git submodule update --init --recursive
	conda env create --file requirements.yml
	conda run --name diffusercam pre-commit install

requirements: requirements.yml
	conda env update --file requirements.yml --prune

