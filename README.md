# GLADD
This is the code for paper "Graph-level Anomaly Detection by Distance From Random Sampling Graph".

## Data Preparation

Some of datasets are put in ./dataset folder. Due to the large file size limitation, some datasets are not uploaded in this project. You may download them from https://chrsmrrs.github.io/datasets/docs/datasets/.

## Requirement

Details in the requirements.txt.

## Train

Run the following code to test on dataset AIDS. For datasets with node attributes, feature chooses default, otherwise deg-num.
If you want to add the test set into the training set as well, append the parameter "--includingTest" to the command. 

	python main.py --datadir dataset\AIDS --DS AIDS --feature default


## Citation
```bibtex
@inproceedings{
}
```
