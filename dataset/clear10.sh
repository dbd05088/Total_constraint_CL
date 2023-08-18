#!/bin/bash

# download and unzip dataset
wget https://clear-challenge.s3.us-east-2.amazonaws.com/clear10-train-image-only.zip
wget https://clear-challenge.s3.us-east-2.amazonaws.com/clear10-test.zip

mkdir clear10
unzip clear10-train-image-only.zip
unzip clear10-test.zip
rm clear10-train-image-only.zip
rm clear10-test.zip
mv train_image_only clear10
mv test clear10
#mv labeled_images/*/*/*.jpg .
echo done
