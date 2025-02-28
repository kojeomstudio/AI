#!/bin/bash

cd /app

pyton -m venv r2r_venv

source r2r_venv/bin/activate

pip install -r ./requirements.txt
python core/r2r_chat_web.py