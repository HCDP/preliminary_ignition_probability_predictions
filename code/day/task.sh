#!/bin/bash

echo "Fetch dependencies, all counties"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/wget_dependencies.py 2024-12-31

echo "Generate API, all counties"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/generate_api.py bi 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/generate_api.py ka 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/generate_api.py mn 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/generate_api.py oa 2024-12-31

echo "Run map script for all counties"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/generate_ignition_map.py bi 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/generate_ignition_map.py ka 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/generate_ignition_map.py mn 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/generate_ignition_map.py oa 2024-12-31

echo "Create statewide mosaic"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/code/day/statewide_mosaic.py