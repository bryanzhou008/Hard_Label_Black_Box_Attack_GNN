# Black-box Node Injection Attack on GNN

### This is the sub-directory of the project by Jingchen Tang
This folder is used to run tests of the single node injection attack on the model 
while allowing edge perturbation on the entire graph. 
The attacker can check the success rate, attack time, other test statistics, 
and whether the nodes injected in the graph was successfully connected into the graph by checking the output of test.py, 
simply by running something like:

`python3 -u test.py --dataset COIL-DEL --model GIN --initialization random --connection random --injection_number 1`

Note: the meanings of some paras are:

`--initialization` : The method of feature initialization on the injected node (random/node_mean)

`--connection` : The method of connection of the injected node into the graph (random/mode)

`--injection_number` : The number of injected nodes
