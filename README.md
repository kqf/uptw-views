# UPTW views dataset [![Build Status](https://travis-ci.com/kqf/uptw-views.svg?token=7bkqqhrPB19pD1YKrAZM&branch=master)](https://travis-ci.com/kqf/uptw-views)


## Exploration
For some insights about the datasets please refer to `exploration/eda.ipynb`. Please install the package before running the notebook:
```python
pip install -e .
```

## Run the model
To run the model for this dataset run this command, however high variance models aren't able to generalize to the new datasets, whereas high bias model perform as good as random choice on training sample.
```bash

# Download the dataset *.csv into ./data folder 
# and then run

make 
```
