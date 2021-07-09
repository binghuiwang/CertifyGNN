# CertifyGNN
Source code of "Certified Robustness of Graph Neural Networks against Adversarial Structural Perturbation", accepted by ACM SIGKDD'21

1. There are two different ways to train the GNN model: 
I) Train on the raw clean graph (train_edge); 
II) Train on the perturbed graph, where the graph is dynamically changed by randomly perturbing the connection status between the edge (train_edge_noise).  

2. The detailed steps are as follows: 
I) Train the GNN model (either train_edge.py or train_edge_noise.py); 
II) Divide the binary graph structure space by running count.py or/and obtain the base probability (P_base) by running threshold.py ; 
III)  Obtain the certified accuracy by running calculate_certify_K.py if not running threshold.py in II), or by running calculate_certify_K_efficient.py where I already stored the P_base.   
 
3. Note that the certified accuracy could be slightly changed due to 1) randomness during the training and 2) random perturbations
 
