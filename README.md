### Project on [Zindi CGAIR crop Yield Estimation](https://zindi.africa/competitions/cgiar-crop-yield-prediction-challenge)

#### Usage:

i). Clone the project

```commandline
git clone https://github.com/RexMudanya/CGIAR-Crop-Yield-Prediction-Challenge.git
```

ii). Navigate into the project 
```commandline
cd CGIAR-Crop-Yield-Prediction-Challenge/
```


iii). Install packages

```commandline
pip install -r requirements.txt
```

iv). Install [jupyter notebooks](https://jupyter.org/install)

```commandline
pip install notebook
```

v). Open terminal/command line (cmd) and create a new directory

```commandline
mkdir data
```

vi). download from data [here](https://zindi.africa/competitions/cgiar-crop-yield-prediction-challenge/data) and place in `data/`

vii). Open terminal/command line (cmd) and run 

```commandline
jupyter notebook
```

viii). Click on relevant notebook, voila!!!

#### Install Hyperopt-sklearn

1. ```commandline 
   git clone https://github.com/hyperopt/hyperopt-sklearn.git
   ```
1. ```commandline 
   cd hyperopt
   ```
1. ```commandline
    pip install -e .
   ```


### Training the Encoder

1. To train encoder `` python train_encoder.py ``
1. Launch tensorboard from terminal to visualize logs `` tensorboard --logdir logs/``
1. Trained encoder saved as `` encoder.h5``

### Folder structure

<pre>
.
├── data
├── images
├── logs
├── models
├── notebooks
├── README.md
├── requirements.txt
├── src
└── submissions
</pre>

Download data [here](https://zindi.africa/competitions/cgiar-crop-yield-prediction-challenge/data)

Check [Leaderboard](https://zindi.africa/competitions/cgiar-crop-yield-prediction-challenge/leaderboard)

Further [Discussions](https://zindi.africa/competitions/cgiar-crop-yield-prediction-challenge/leaderboard)