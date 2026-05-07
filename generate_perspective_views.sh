#!/bin/bash

for ((i=-180; i<=180; i+=10))
do
    python3 kadai2.py input.jpg $i
done