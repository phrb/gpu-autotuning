GraphGen generates a collection of labeled, undirected and connected graphs. 
The datasets generated can be used for the performance evaluation of 
frequent subgraph mining algorithms and graph query processing algorithms. 
The generator is based on the IBM Quest Synthetic Data Generation Code for 
Associations and Sequential Patterns 
(http://www.almaden.ibm.com/cs/projects/iis/hdb/Projects/data_mining/datasets/syndata.html#assocSynData). 



[Options]
-ngraphs: 
The number of graphs in the dataset (in 1000's) (default: 10)

-size: 
The averaged size (the number of edges) of each graph (default: 20)
The size is a normal distribution with the input as the mean and 5 as the variance.

-nnodel: 
The number of unique node labels (default: 20)

-nedgel:
The number of unique edge labels (default: 20)

-density:
The averaged density of each graph ([0, 1], default: 0.3)
The density, d, of a graph is defined as the number of edges in the graph divided by the number of edges in a complete graph, 
i.e., d = |E|/(|V|(|V|-1)/2).
The density is a normal distribution with the input as the mean and 0.01 as the variance.

-nedges:
The number of unique edges in the whole dataset (default: 100)
A unique edge is defined as a 3-tuple (u_label, e_label, v_label), 
where u_label is the label of a node u, e_label is the label of the edge, and v_label is the label of a node v.

-edger:
The averaged edge ratio of each graph ([0, 1], default: 0.2)
The edge ratio of a graph is defined as the number of unique edges in the graph divided by the number of edges in the graph.
The edge ratio is a normal distribution with the input as the mean and 0.01 as the variance.

-fname:
The generated dataset is written to a disk file filename.data

-randseed: 
The random seed used to generate the transaction graphs (must be negative)



[Usage of the generator]
GraphGen [options]
GraphGen -help

[Output format]
An ascii file is outputted with the following data format (see the example output file, test.data)
Each graph is descried as follows:
"t # N" means the Nth graph (N starts from 0),
"v M L" means that the Mth vertex in this graph has label L (M and L start from 0),
"e P Q L" means that there is an edge connecting the Pth vertex with the Qth vertex. 
The edge has label L (L starts from 0. The edge label is the node label is independent).


The executable runs in a Unix machine. The source codes are available upon request. If you wish to cite the generator in your paper, you may cite it as follows:

James Cheng, Yiping Ke, and Wilfred Ng. GraphGen: A graph synthetic generator. http://www.cse.ust.hk/graphgen/, 2006.


This generator was first used in 

J. Cheng, Y. Ke, W. Ng, and A. Lu. FG-Index: Towards Verification-Free Query Processing on Graph Databases. 
To appear in Proceedings of the 26th ACM SIGMOD International Conference on Management of Data (SIGMOD), 2007. 


[License Agreement]
Please read this agreement carefully before using this software. By using the software, you agree to become bound by the terms of this license. If you do not agree to the terms of this license, please delete this package and DO NOT USE it.

1. This copy is for your internal use only. Please DO NOT re-distribute it without the permission from the authors (James Cheng, Yiping Ke, and Wilfred Ng). Please DO NOT post this package on the internet.
2. This software should not be used for any commercial interest.
3. The authors do not hold any responsibility for the correctness of this generator. 


Your feedback and test results are welcome. 

Contacts: 
James Cheng (csjames@cse.ust.hk)
Yiping Ke (keyiping@cse.ust.hk)
Wilfred Ng (wilfred@cse.ust.hk)


