# Pure Transformers Can Be Powerful Hypergraph Learners
TokenHGT: Pure Transformers Can Be Powerful Hypergraph Learners

This is my master thesis project in [DIAG](http://www.diag.uniroma1.it/en), [Sapienza University of Rome](https://www.uniroma1.it/en/pagina-strutturale/home).

![sapienza-big](https://user-images.githubusercontent.com/24941293/152373391-ac062aac-750a-45cd-bf40-9851cf2911f1.png)


Author: [Kai Peng](https://github.com/TuDou-PK)   Acdamic Year: 2022/2023

Thesis supervisor:
- Research Fellow: [Giovanni Trappolini](https://sites.google.com/view/giovannitrappolini)
- Full Professor: [Fabrizio Silvestri](https://sites.google.com/diag.uniroma1.it/fabriziosilvestri/home?authuser=0)

# Quick Facts

Existing Problems:
- Graph/hypergraph convolution operations (message-passing methods) can lead to **over-smoothing** problems.
- Modified structure transformers are designed for **specific tasks may limit versatility**, hindering integration into multi-task and multi-modal general-purpose attentional architectures.
- [**Tokenized Graph Transformer(TokenGT)**](https://github.com/jw9730/tokengt) has successfully addressed these issues in the graph area. But not address these issues in the hypergraph area.

Thesis contributions:
- This thesis aims to **expand TokenGT to the hypergraph area** to solve the limitations of message-passing and graph-specific structural modifications in the hypergraph field.
- Provide an optional method for processing hypergraphs.

# TokenHGT: Based on TokenGT
This work is based on tokenGT, our model called Tokenized HyperGraph Transformer(TokenHGT), but because hypergraphs are different from graphs, there are still innovations in our pipeline. 

The following is a comparison between tokenGT and tokenHGT pipelines. The tokenGT pipeline as follows:

![image](https://github.com/TuDou-PK/TokenHGT/assets/24941293/68b94594-067d-423a-b639-9ef735ec5324)

Our model TokenHGT pipeline as follows:

![image](https://github.com/TuDou-PK/TokenHGT/assets/24941293/90a2f2ad-467c-407b-bf28-f52d6c840ee9)

The differences such as: 
- The laplacian eigendecomposition formula is different.
- Each graph edge only contains 2 nodes, each hyperedge contains a different number of nodes, so we can't do eigenvector alignment by the number of connected nodes as graph edges do, so I directly add eigrnvectors for feature fusion.
- I concat feature tokens with eigenvector tokens instead of summing them, according to experiment result....
- I didn't use "Type Identifier", it will reduce the performance of the model(In my personal opinion, it's not human-made features, it's noise).
