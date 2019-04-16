# Hash code indexing for cross-modal retrieval 
###### Read me ######
We have provided code for 4 steps on MIRFlickr-SePH dataset.
1. Compute cluster member and count member (both modality image and text).
2. Compute cluster probability (both query and training).
3. Train model and test (cross-modal including, x= image modality and y=text modality).
4. Evaluate modal
	(4.1) Computer exhaustive search MAP with hamming distance.
	(4.2) Computer MAP@R R=1 to 50.
	(4.3) Computer MAP ranking and no-ranking in clusters (We can set how many clusters we want to compute).
	(4.4) Computer MAP mixed candidate with ranking and no-ranking.
5. This link for download input files.
https://drive.google.com/drive/folders/19IxJeDAgx8rV7CjlcoUqOtlUEIf3jY21
There are 3 folders, including the input folder, measure folder, and output folder. There are 3 files in the input folder, the first file is full 64-bit, second file is index 8-bit and third file is raw query for training and testing. 
### If you have any questions or found any problems, please don't hesitate to ask Sarawut Markchit smarkchit@gmail.com or Prof. Chih-Yi Chiu cychiu@mail.ncyu.edu.tw
