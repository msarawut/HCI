# HCI
Hash code indexing
###### Read me ######
We provide code for 4 steps.
Note that before run a code please create 3 floders input, measure, and output. Then put all input files in input floder.
1. Compute cluster member and count member (both modality image and text).
2. Compute cluster probability (both query and training).
3. Train model and test (train cross-modat x= image modality, y=text modality).
4. Evaluate
	(4.1) Computer exhaustive search MAP with hamming distance.
	(4.2) Computer MAP@R R=1 to 50.
	(4.3) Computer MAP ranking and no-ranking in clusters (We can set how many clusters we want to compute.)
	(4.4) Computer MAP mixed candidate with ranking and no-ranking.
5. This link for download input files.
https://drive.google.com/drive/folders/19IxJeDAgx8rV7CjlcoUqOtlUEIf3jY21
There are 3 flies. First flie is full 64-bit. Second file is index bit (example for 8-bit). Thrid file is raw query for traininf and testing.
