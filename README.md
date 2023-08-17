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
- This thesis aims to expand TokenGT to the hypergraph area to solve the limitations of message-passing and graph-specific structural modifications in the hypergraph field.
- Provide an optional method for processing hypergraphs.
