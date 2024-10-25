#!/bin/bash -x

celery -A main worker --loglevel=info --max-tasks-per-child=20
