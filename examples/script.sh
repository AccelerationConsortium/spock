#!/bin/bash

# Create the main directory
mkdir -p db

# Loop to create subdirectories
for i in {0..50}
do
  mkdir -p db/db$i
done

echo "Directories created successfully."
