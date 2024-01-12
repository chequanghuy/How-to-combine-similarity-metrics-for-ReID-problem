# -HOW-TO-COMBINE-SIMILARITY-METRICS-FOR-RE-IDENTIFICATION-PROBLEM-

## Introduction


## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- [pytorch>=0.4](https://pytorch.org/)
- torchvision
- [ignite=0.1.2](https://github.com/pytorch/ignite) 
- [yacs](https://github.com/rbgirshick/yacs)

## Test

```bash
python3 tools/test.py --config_file='configs/market1501.yml' --metric='cs+ct' --coeff_method='logistic' --rand_seed 0 --k 5 --n_data 1000 
```
#### Arguments:
- `--config_file`: Path to the config file.
- `--metric`: Choose the metric in ["cosine", "centroid", "cs+ct"]. Default is "cosine".
- `--all_cameras`: Considering all cameras. (Optional)
- `--uncertainty`: Enable uncertain centroid calculation. (Optional)
- `--weighted`: Use weighted centroid calculation when uncertainty is provided. (Optional)
- `--k`: Top-k similarity based on uncertainty. Default is 5.
- `--vis_top`: This argument determines the number of top_k values to be visualized. Default is 0.
- `--coeff_method`: Specify the method to be used for training. Options are "logistic" or "svm". Default is "logistic".
- `--rand_seed`: Set a specific random seed for reproducibility. Default is 0.




