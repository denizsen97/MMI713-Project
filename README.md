# MMI713-Project
My implementation of parallelized weighted TrustRank on subreddits using CUDA, Thrust and OpenMP APIs

The code can be tried from the Google Colab notebook:
https://colab.research.google.com/drive/1tbKozogge9oJjrxzmV6B4j6OUypRS3ri?usp=sharing


It requires SNAP: Social Network: Reddit Hyperlink dataset to work on. The data can be fetched from https://snap.stanford.edu/data/soc-RedditHyperlinks.html


To compile, run the command below from CMD: 
- nvcc -Xcompiler -fopenmp -o kernel kernel.cu

Put the soc-redditHyperlinks-title.tsv and soc-redditHyperlinks-title.tsv files inside the same directory as the binary file without changing their names.

To run the program:

./kernel runtype epsilon beta augmentation trusted_page_number gpu_run_type

- runtype: "device" to run in gpu or "host" to run in CPU

- epsilon: a float between 0 and 1 such as 0.001 for the error amount to stop the power iteration

- beta   : a float between 0 and 1 such as 0.8 for the teleportation probability 

- augmentation: an integer greater than or equal to 0, to indicate the augmentation amount of the read data

- trusted_page_number: number of trusted pages to gather from the first PageRank

- gpu_run_type: "thrust" for Thrust API or "reduce" for custom reduction kernel to make the error calculation
