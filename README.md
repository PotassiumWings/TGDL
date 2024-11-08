# TGDL

This is the official implementation of TGDL.

# Requirements

Python 3.9

python -r requirements.py

# Model Training

```
python start.py Model Dataset NumOfSubgraphs Seed [Additive parameter settings]
```

Following are examples of using TGDL:

```
python start.py STGCN NYCBike 1 0 "--num-epoches 1"
python start.py GraphWavenet NYCBike 6 0
python start.py MTGNN NYCTaxi 8 1
python start.py MSDR PEMSD7 6 12345 "--compare --loss-1 0"
```