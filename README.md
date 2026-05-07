# FML-Activity-Classification
Federated Machine Learning application for Human Activity Recognition (UCI HAR). Developed as part of the FML Programmierpraktikum at TU Berlin (SBE Group).
Topic B

## Dependency Management
This repository uses [uv](https://docs.astral.sh/uv/) for dependency management. Please install `uv` in your local shell then run: 
```bash
uv sync
```

This will create a virtual environment (.venv) with the specified python version and package versions, so we all develop with the same packages and don't run into any conflicts. To use the new environment/kernel in VS-Code, press `cmd-shift-P` (on mac, on windows `ctrl-shift-P`) and select a Python interpreter path. Since the jupyter notebooks are in the notebooks directory and not in the main project directory with the config files, you have to tell vscode to look there, so for the interpreter path enter `<your_local_path_to_FML-Activity-Classification>/.venv/bin/python`. Then click on Select kernel and choose the `fml-activity-classification` kernel.


## Project Organization
**Communication:** Discord, WhatsApp
**Task Management:** GitHub Projects --> agile Approach

## Meetings
Weekly meetings on Friday to catch up with the team
--> documentation with md



# Team Roles
**Data Engineering & Partitioning:**
--> Python pipeline to load and preprocess the dataset

**ML Architecture:**
--> PyTorch neural network for activity classification

**FL Communication:**
--> Developing model serialization and preparing interfaces for FL

**Infrastructure & Dashboard:**
--> designing the infrastucture around the project


