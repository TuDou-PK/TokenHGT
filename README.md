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




# Experiment
Now let's do some experiments!

The tokenHGT algorithm is designed to operate at the graph level, making it suitable for datasets that contain a significant number of hypergraphs. Ideally, the dataset should include both node and hyperedge features to capture the structural and attribute information inherent in the hypergraphs.

However, it is challenging to find readily available datasets that meet these requirements. Therefore, we have explored two methods to create suitable hypergraph datasets.

I find two choices:
- Cvonvert a graph into hypergraph, using Dual Hypergraph Transformation (DHT) from [Edge Representation Learning with Hypergraphs](https://github.com/harryjo97/EHGNN).
- Convert a text into hypergraph from [Hypergraph Attention Networks for Inductive Text Classification](https://github.com/kaize0409/HyperGAT_TextClassification)

I tried DHT method in PCQM4Mv2 and ogbg-molhiv, that's interesting to convert a molecular graph to a hypergraph, but due to device limitation(Money is all you need :), I have to give up it.

Convert a text to a hypergraph is a good chioce, the dataset alway small~ More detials about [how to convert a text into a hypergraph](https://github.com/kaize0409/HyperGAT_TextClassification).

Here is the dataset I used:

![image](https://github.com/TuDou-PK/TokenHGT/assets/24941293/05b22d98-7d79-4396-979b-08a8614e1023)

Result:

![image](https://github.com/TuDou-PK/TokenHGT/assets/24941293/bf4d91d7-084d-4537-9e74-b148ffe1dcaa)

You will find I used 5 datasets but only show 3 dataset result, that's because our model performance too terrible on 20NG & Ohsumed.

# Conclutions & Limitations
Conclusions:
- The tokenHGT model generally applies pure transformers to the hypergraph area.
- The tokenHGT’s effectiveness in overcoming the limitations of message passing methods, leading to superior performance on specific datasets. 
- Meanwhile, the pure transformer architecture guarantees the versatility of the models, which contributes to future multimodal research.

Limitations:
- TokenHGT is not good at processing large hypergraphs. According to Graphomer, the self-attention module exhibits a quadratic complexity, which poses limitations on its applicability to large graphs.
- It requires a suitable hypergraph dataset, which can be challenging to find.

# Code Introduction

The full code is in "FullCode" file, keep the file structure.

- Download row MR dataset: MR_Download.py
- Generate LDA file: generate_lda.py, detials [Here](https://github.com/kaize0409/HyperGAT_TextClassification)
- Run main code: main.py
  *  The model structure in: model_mr.py
  *  process the LDA data: preprocess.py & utils.py
  *  Detials about hypergraph laplacian eigendecomposition: eigen.py
