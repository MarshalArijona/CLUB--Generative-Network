# CLUB--Generative-Network
# Link
[Technical Report Link](https://www.researchgate.net/publication/349881819_Variational_Contrastive_Log_Ratio_Upper_Bound_of_Mutual_Information_for_Training_Generative_Model)
# Abstract
Abstract—This research is about the application of mutual
information estimator for generative models. This research applies variational contrastive log-ratio upper bound (vCLUB)
minimization algorithm to minimize the mutual information
between mixture distribution between real data distribution and
generated data distribution and binary distribution that alternate
between real data distribution and generated data distribution.
The aim is the same as minimizing Jensen-Shannon divergence
between real data distribution and generated data distribution
which is the purpose of generative adversarial network. Furthermore, this research proposed two MI-based generative models,
CLUB-sampling generative network (vCLUB-sampling GN) and
vCLUB-non sampling generative network (vCLUB-non sampling
GN). Both models are developed as deep neural networks. Result
of experiments show that vCLUB-non sampling generate better
samples than vCLUB-non sampling GN and variational L1-out
generative network (vL1-out GN). Unfortunately, both vCLUB
GN models are outperformed by generative adversarial network
(GAN).

# Algorithm
![alt text](https://github.com/MarshalArijona/CLUB--Generative-Network/blob/main/algorithm.PNG?raw=true)

# Result
![alt text](https://github.com/MarshalArijona/CLUB--Generative-Network/blob/main/generated_CLGAN_non_sampling.PNG?raw=true)
![alt text](https://github.com/MarshalArijona/CLUB--Generative-Network/blob/main/score.PNG?raw=true)
