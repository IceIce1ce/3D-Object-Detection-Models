#!/bin/bash

echo "--- 'for' loop over array elements ---"

# Define an array
my_array=("apple pie"
"banana split"
"cherry tart"
"orange juice")

echo "Iterating through array elements:"
for dessert in "${my_array[@]}"; do
  echo "Enjoying: $dessert"
done


# You can also iterate through array indices if you need the index
echo "Iterating through array indices and values:"
for index in "${!my_array[@]}"; do
  echo "Index: $index, Value: ${my_array[$index]}"
done