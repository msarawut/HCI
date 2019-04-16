# Hash code indexing for cross-modal retrieval 
### Read me 
We have provided code for 4 steps on MIRFlickr-SePH dataset (we employed the SePH hashing method to generate MIRFlickr dataset to binary code).
1. Compute cluster member and count member (both modality image and text).
2. Compute cluster probability (both query and training).
3. Train model and test (cross-modal including, x= image modality and y=text modality).
4. Evaluate modal
	(4.1) Compute exhaustive search MAP with hamming distance.
	(4.2) Compute MAP@R R=1 to 50.
	(4.3) Compute MAP ranking and no-ranking in clusters (We can set how many clusters we want to compute).
	(4.4) Compute MAP mixed candidate with ranking and no-ranking.
5. This link for download input files.
https://drive.google.com/drive/folders/19IxJeDAgx8rV7CjlcoUqOtlUEIf3jY21
There are 3 folders, including the input folder, measure folder, and output folder. There are 3 files in the input folder, the first file is full 64-bit, second file is index 8-bit and third file is raw query for training and testing. 

The MIRFlickr dataset is a cross-modal dataset. It has 25000 instances collected from the Flickr website. 
Each instance consists of an image, associated textual tag, and one or more of 24 predefined semantic labels. 
(M. J. Huiskes, M. S. Lew, The MIR flickr retrieval evaluation, In Proceedings of the 1st ACM international conference on Multimedia information retrieval 2008 Oct 30 (pp. 39-43). ACM.)

Semantics-preserving hashing (SePH) is a cross-modal hashing method. 
It transformed the given semantic affinities of training data to a probability distribution and approximates it with another one in Hamming space, via minimizing their Kullback-Leibler divergence. 
SePH used any kind of predictive models such as linear ridge regression, logistic regression, or kernel logistic regression as hash functions to learn in each view for projecting the corresponding view-specific features into hash codes. 
(Z. Lin, G. Ding, J. Han, J. Wang, Cross-view retrieval via probability-based semantics-preserving hashing, IEEE transactions on cybernetics. 2017 Dec;47(12):4342-55.)

#### If you have any questions or found any problems, please don't hesitate to ask Sarawut Markchit smarkchit@gmail.com or Prof. Chih-Yi Chiu cychiu@mail.ncyu.edu.tw
