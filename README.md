# Clustering cell types in the post-mortem middle temporal gyrus using snRNA-seq data
02-620 Machine Learning for Scientists

We aimed to cluster cell types using single nuclear RNA-seq (snRNA-seq) data from the Middle Temporal Gyrus (MTG), using a publicly available dataset found on Allen Brain Atlas.

Methods:
1. PCA is done to reduce the dimensionality of the data using Scikit-learn.
2. Two types of clustering are implemented: Louvain clustering (coded from scratch using NetworkX) and Gaussian Mixture Models (used Scikit-learn).
3. Marker genes are annotated from Louvain clustering, with adapted criteria from Hodge et al.
4. A Naive Bayes classifier is implemented from scratch and is trained on both types of clustering for comparison, with a train-test split of 50-50.
5. Clustering is evaluated using Dunn Index (DI) and Silhouette Coefficient (SC), both coded from scratch.
6. The classifier is evaluated using a confusion matrix (All-vs-All) and ROC Curve (One-vs-Rest).

Results:
1. PCA was most time-consuming step, and even with 2000 PCs, only aboout 33% of variance could be captured. 20 PCs are used for downstream tasks.
2. Louvain and GMM clustering are compared on UMAP, which showed that Louvain gave more clusters than GMM.
3. Marker genes were found corresponding to cell types in the brain and validated as cluster-specific via violin plots.
4. Classifier gave 70-80% accuracy using both types of clustering.
5. GMM had higher values for both DI and SC, which suggested it was a better form of clustering.
6. Confusion matrix showed better accuracy when trained on GMM than Louvain clustering. ROC Curve showed higher AUC for Louvain clustering.

Data availability:
For an in-depth explanation of the project, the code, outputs and report are made available in the repository.
