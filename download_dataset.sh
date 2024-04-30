#!/bin/bash

kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip
python3 prepare_dataset.py