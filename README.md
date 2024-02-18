# GLADD
This is the code for paper "Multi-representations Space Separation based Graph-level Anomaly-aware Detection".

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
@inproceedings{Lin_2023,
   series={SSDBM 2023},
   title={Multi-representations Space Separation based Graph-level Anomaly-aware Detection},
   url={http://dx.doi.org/10.1145/3603719.3603739},
   DOI={10.1145/3603719.3603739},
   booktitle={35th International Conference on Scientific and Statistical Database Management},
   publisher={ACM},
   author={Lin, Fu and Gong, Haonan and Li, Mingkang and Wang, Zitong and Zhang, Yue and Luo, Xuexiong},
   year={2023},
   month=jul,
   collection={SSDBM 2023}
}
```
