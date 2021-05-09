# TrafficStream

Code for **TrafficStream: A Streaming Traffic Flow Forecasting FrameworkBased on Graph Neural Networks and Continual Learning**（IJCAI 2021). TrafficStream is a streaming traffic flow forecasting framework based on Graph Neural Networks (GNNs) and Continual Learning (CL), achieving accurate predictions and high efficiency.

### Requirements

* python = 3.8.5
* pytorch = 1.7.1
* torch-geometric = 1.6.3

```
conda env create -f trafficStream.yaml
```
  
### Data

Download raw data from [this](https://drive.google.com/file/d/1P5wowSaNSWBNCK3mQwESp-G2zsutXc5S/view?usp=sharing), unzip the file and put it in the `data` folder

### Usages

* TrafficStream-STModel
```
python main.py --conf conf/trafficStream.json --gpuid 1
```
* Expansible-STModel (lower bound):
```
python main.py --conf conf/expansible.json --gpuid 1
```
* Retrained-STModel (upper bound):
```
python main.py --conf conf/retrained.json --gpuid 1
```
* Static-STModel:
```
python main.py --conf conf/static.json --gpuid 1
```

