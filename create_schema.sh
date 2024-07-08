#!/bin/sh

# Display all env variable
source .env

echo 'Creating schema'
python3 src/download/create_schema.py
