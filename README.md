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
python tools/test.py --config_file "configs/market1501.yml" --metric='cs+ct'
```
#### Arguments:
- `--config_file`: Path to the config file.
- `--metric`: Choose the metric in ["cosine", "centroid", "cs+ct"]. Default is "cosine".
- `--all_cameras`: Considering all cameras. (Optional)
- `--uncertainty`: Enable uncertain centroid calculation. (Optional)
- `--weighted`: Use weighted centroid calculation when uncertainty is provided. (Optional)
- `--k`: Top-k similarity based on uncertainty. Default is 5.
