# Multi Node Injection Attack

This folder maily contains the modified source code for Multi-Node Injection Attack on GNN. Most modifications are in:

Sign_Opt.py: contains modifications to 

  (1): Remove initial search; 
  
  (2): Restrain update in theta matrix only related to injected nodes
  
  (3): Node feature Initialization; 
  
  (4): Node connection initialization; 
 
 test.py: Additional metrics and args for stats related to our experiments; some args include:
 
  (1): --node_injection: store true, add to use node injection attack
  
  (2): --injection_percentage: upper bound of number of nodes to inject
  
  (3): --iterative: stroe true, whether or not to inject 1 node, 2 nodes, ... , upper limit nodes
  
  (4): --initialization: node feature initialization method
  
  (5): --connection: connection initialization method
