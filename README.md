# Sentiment Analysis on Movie Reviews

## Overview

This project aims to use NLP techniques to analyze the sentiment of a given movie review based on the reviews it has learned from.

## Dataset

The dataset used is the IMDB Dataset of 50K Movie Reviews.

If you would like to use your own dataset, change the DATASET_PATH to the csv file of the dataset you would like to use.

## Interactive File

The [interactive python file](./interactive.py) walks through the whole process and allows you to make choices at each point which has an effect on the final accuracy of the model. Different choices can lead to higher or lower accuracies. Therefore, this python file can be used to figure out what works best in different scenarios (Ex. with a smaller dataset, or with reviews that haven't been preprocessed, etc...)

## Usage

The imported libraries must be installed before being able to run any of the files. This can be done using the command: `pip install -r requirements.txt`
