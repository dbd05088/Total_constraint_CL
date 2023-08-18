#!/bin/bash

# download and unzip dataset
wget https://clear-challenge.s3.us-east-2.amazonaws.com/clear100-train-image-only.zip
wget https://clear-challenge.s3.us-east-2.amazonaws.com/clear100-test.zip

mkdir clear100
unzip clear100-train-image-only.zip
unzip clear100-test.zip
rm clear100-train-image-only.zip
rm clear100-test.zip
mv train_image_only clear100
mv test clear100
echo done
