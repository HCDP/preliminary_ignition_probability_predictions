#!/bin/bash

echo "[task.sh] [1/7] Starting Execution."
export TZ="HST"
echo "It is currently $(date)."
if [ $CUSTOM_DATE ]; then
    echo "An aggregation date was provided by the environment."
else
    export CUSTOM_DATE=$(date -d "1 day ago" --iso-8601)
    echo "No aggregation date was provided by the environment. Defaulting to yesterday."
fi
echo "Aggregation date is: " $CUSTOM_DATE
source envs/prod.env

echo "[task.sh] [2/7] Fetching dependencies, all counties."
python3 -W ignore /code/wget_dependencies.py $CUSTOM_DATE

echo "[task.sh] [3/7] Generating API, all counties."
python3 -W ignore /code/generate_api.py bi $CUSTOM_DATE
python3 -W ignore /code/generate_api.py ka $CUSTOM_DATE
python3 -W ignore /code/generate_api.py mn $CUSTOM_DATE
python3 -W ignore /code/generate_api.py oa $CUSTOM_DATE

echo "[task.sh] [4/7] Running map workflow, all counties"
python3 -W ignore /code/generate_ignition_map.py bi $CUSTOM_DATE
python3 -W ignore /code/generate_ignition_map.py ka $CUSTOM_DATE
python3 -W ignore /code/generate_ignition_map.py mn $CUSTOM_DATE
python3 -W ignore /code/generate_ignition_map.py oa $CUSTOM_DATE

echo "[task.sh] [5/7] Creating statewide mosaic."
python3 -W ignore /code/statewide_mosaic.py

echo "[task.sh] [6/7] Preparing upload config."
cd /sync
python3 inject_upload_config.py config.json $CUSTOM_DATE

echo "[task.sh] [7/7] Uploading data."
python3 upload.py

echo "[task.sh] All done!"