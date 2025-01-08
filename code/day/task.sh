#!/bin/bash

echo "Fetch dependencies, all counties"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/day/code/wget_dependencies.py 2024-12-31

echo "Run map script for all counties"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/day/code/generate_ignition_map.py bi 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/day/code/generate_ignition_map.py ka 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/day/code/generate_ignition_map.py mn 2024-12-31
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/day/code/generate_ignition_map.py oa 2024-12-31

echo "Create statewide mosaic"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/ignition_prob/day/code/statewide_mosaic.py