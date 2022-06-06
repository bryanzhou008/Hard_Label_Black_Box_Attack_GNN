# Single Node Injection

This folder maily contains the modified source code for Single Node Injection on GNN. Most modifications are in:

test.py: Additional metrics and args for stats related to our experiments; some args include:

(1): --node_number: number of nodes to be injected

(2): --injection_percentage: upper bound of number of nodes to inject

(3): --initialization: node feature initialization method

(4): --connection: connection initialization method


# Visualization and Demo

visualization.py: Visualizations of the adversiarial attack process. Generation results are stored in:

(1): test_multi_node_mode

(2): test_multi_node_random

(3): test_single_node

demo.ipynb: Our demo mentioned in paper Appendix 7.1
