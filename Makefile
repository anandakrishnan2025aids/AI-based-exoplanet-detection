.PHONY: help run-pipeline-podman run-pipeline-python install-dev

SHELL := /bin/bash

# Default values for pipeline run, can be overridden from command line
# e.g. make run-pipeline-python TICS_TBL_FP=/my/data/tics.csv

INPUTS_DIR ?= $(shell pwd)/inputs
RUN_DIR ?= $(shell pwd)/run/$(shell date +%Y%m%d_%H%M%S)
TICS_TBL_FN ?= tics_tbl.csv
TICS_TBL_FP ?= $(INPUTS_DIR)/$(TICS_TBL_FN)
DATA_COLLECTION_MODE ?= 2min
NUM_PROCESSES ?= 1
NUM_JOBS ?= 1
DOWNLOAD_SPOC_DATA_PRODUCTS ?= false
EXTERNAL_DATA_REPOSITORY ?= null
STELLAR_PARAMETERS_SOURCE ?= ticv8
RUWE_SOURCE ?= gaiadr2
EXOMINER_MODEL ?= exominer++_single

help:
	@echo "Available commands:"
	@echo "  install-dev          - Create a conda environment for development from arm64 architecture."
	@echo "  run-pipeline-python  - Run the ExoMiner pipeline using Python."
	@echo "  run-pipeline-podman  - Run the ExoMiner pipeline using Podman."
	@echo ""
	@echo "You can override variables, e.g. 'make run-pipeline-python TICS_TBL_FP=/path/to/your/tics.csv'"

install-dev:
	@# Detect architecture and choose the appropriate conda environment file
	@ARCH=$(shell uname -m)
	@CONDA_ENV_FILE=""
	@if [ "$$ARCH" = "arm64" ] || [ "$$ARCH" = "aarch64" ]; then \
		CONDA_ENV_FILE="exominer_pipeline/conda_env_exoplnt_dl_arm64.yml"; \
		echo "Detected ARM64 architecture. Creating conda environment 'exoplnt_dl' from $$CONDA_ENV_FILE..."; \
	elif [ "$$ARCH" = "x86_64" ]; then \
		CONDA_ENV_FILE="exominer_pipeline/conda_env_exoplnt_dl_amd64.yml"; \
		echo "Detected AMD64/x86_64 architecture. Creating conda environment 'exoplnt_dl' from $$CONDA_ENV_FILE..."; \
	else \
		echo "Unsupported architecture: $$ARCH. Please manually specify the correct conda environment file."; \
		exit 1; \
	fi; \
	if [ -f "$$CONDA_ENV_FILE" ]; then \
		conda env create -f $$CONDA_ENV_FILE; \
	fi
	@echo "To activate the environment, run: conda activate exoplnt_dl"

run-pipeline-podman:
	@echo "Running ExoMiner pipeline with Podman..."
	exominer_pipeline/run_podman_application.sh --exominer_pipeline_run_dir $(RUN_DIR) --tics_tbl_fp $(TICS_TBL_FP)

run-pipeline-python:
	@echo "Running ExoMiner pipeline with Python..."
	python exominer_pipeline/run_pipeline.py \
		--output_dir $(RUN_DIR) \
		--tic_ids_fp $(TICS_TBL_FP) \
		--data_collection_mode $(DATA_COLLECTION_MODE) \
		--num_processes $(NUM_PROCESSES) \
		--num_jobs $(NUM_JOBS) \
		--download_spoc_data_products $(DOWNLOAD_SPOC_DATA_PRODUCTS) \
		--external_data_repository $(EXTERNAL_DATA_REPOSITORY) \
		--stellar_parameters_source $(STELLAR_PARAMETERS_SOURCE) \
		--ruwe_source $(RUWE_SOURCE) \
		--exominer_model $(EXOMINER_MODEL)