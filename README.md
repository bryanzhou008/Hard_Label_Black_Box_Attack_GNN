# Hard Label Black Box Node Injection Attack on Graph Neural Network
This is the official Pytorch code repository of the paper "Hard Label Black Box Node Injection Attack on Graph Neural Network". 

Detailed descriptions can be found in the separate folders.


## General Instructions adapted from [CCS21_GNNattack](https://github.com/mujm/CCS21_GNNattack)

### Black-box Adversarial Attack to GNN

Code for [A Hard Label Black-box Adversarial Attack Against Graph Neural Network](https://arxiv.org/pdf/2108.09513)

For questions/concerns/bugs please feel free to email [mujm19@mails.tsinghua.edu.cn](mujm19@mails.tsinghua.edu.cn)

This code was tested with python 3.6 and pytorch 1.4

#### Train target models
To train target GNN models (GIN, SAG, GUNet) on three datasets (COIL-DEL, IMDB-BINARY, NCI1), use [main.py](https://github.com/mujm/Black-box-Adversarial-Attack-to-GNN/blob/master/main.py)

e.g., to train GIN model on NCI1 dataset, use:

`python -u main.py --dataset NCI1 --model GIN`

the trained model will be saved in path: `./trained_model` (create by yourself if does not exist).

#### Attack target GNN
To attack target GNN models via our hard lable black-box adversarial attack, use [test.py](https://github.com/mujm/Black-box-Adversarial-Attack-to-GNN/blob/master/test.py)

e.g., to attack SAG model on IMDB-BINARY dataset (make sure you have trained the model before attack), use:

`python -u test.py --dataset IMDB-BINARY --model SAG`

the attack results will be saved in path: `./out1`

Note: the meanings of some paras are:

`--effective` : whether or not use QEGC algorithm

`--id` : the id of this trail (10 trails in total)

`--search` : the search strategy of CGS algorithm

##### process the attack results

To process the attack results, use [process_results.py](https://github.com/mujm/Black-box-Adversarial-Attack-to-GNN/blob/master/process_results.py)
It will print the SR/AP/AQ/QT/APR(ignore)/AD(ignore) of one trail.

e.g., run: 

`python -u process_results.py --dataset IMDB-BINARY --id 5 --effective 0`

it will print the SR/AP/AQ/AT of our and random attacks against GIN model which is trained on IMDB-BINARY dataset.

Note: the attack results of two GNN models (e.g., GIN and SAG) will overlap so make sure you save them in different folders.

#### Random attack
To attack target GNN models via random attack, use [random_attack.py](https://github.com/mujm/Black-box-Adversarial-Attack-to-GNN/blob/master/random_attack.py)

e.g., to attack GUNet model on IMDB-BINARY dataset (make sure you have trained the model before attack), use:

`python -u random_attack.py --dataset IMDB-BINARY --model GUN`

the attack results will be saved in path: `./out1`

#### RL-S2V attack
please refer the initial paper and code of [RL-S2V](https://github.com/Hanjun-Dai/graph_adversarial_attack)

#### Detect Adversarial Graphs

The first countmeasure against our attack: detection.

##### generate training dataset for detectors

To generate training dataset for binary detectors, use [detection_train_dataset.py](https://github.com/mujm/Black-box-Adversarial-Attack-to-GNN/blob/master/detection_train_dataset.py). It will randomly select some graphs in the original training dataset and acquire their corresponding adversarial graphs via our attack.

e.g., to get training dateset for GIN detector (i.e., deploy GIN as the binary detector) on COIL-DEL dataset, run:

`python -u detection_train_dataset.py --dataset COIL-DEL`

the generated training dataset will be saved in folder: `./detection` (create by yourself if it does not exist)

Note: the testing dataset is the adversarial graphs and there corresponding normal graphs in [## Attack target GNN]

To generate training dataset with PGD attack, please refer to the initial paper and code of [PGD](https://github.com/KaidiXu/GCN_ADV_Train)
#### train and test detectors

To train detectors and to evaluate their effectiveness, use [binary_detect.py](https://github.com/mujm/Black-box-Adversarial-Attack-to-GNN/blob/master/binary_detect.py)
e.g., to train a GIN detector with training dataset generated by our attack and to evaluate its performance, run:

`python -u binary_detect.py --method Our --dataset COIL-DEL --model GIN`

Note:  `--method` : the attack method to generate training dataset for the detector. ['Our' or 'Typo']

`--model` : the model structure of detector


