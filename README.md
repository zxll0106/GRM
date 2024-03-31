# Learning Group Residual Representation for Group Activity Prediction (ICME 2023)
This repo contains code of our paper "Learning Group Residual Representation for Group Activity Prediction". 

_Xiaolin Zhai, Zhengxi Hu, Dingye Yang, Shichao Wu and Jingtai Liu_

[[paper](https://ieeexplore.ieee.org/abstract/document/10220015)]

## Dependencies

- Software Environment: Linux 
- Hardware Environment: NVIDIA RTX 3090
- Python `3.8`
- PyTorch `1.11.0`, Torchvision `0.12.0`
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)

## Prepare Datasets

1. Download publicly available datasets from following links: [Volleyball dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and [Collective Activity dataset](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip).
2. Unzip the dataset file into `data/volleyball` or `data/collective`.


## Train
1. **Train the Base Model**: Fine-tune the base model for the dataset. 
```shell
    # Volleyball dataset
    cd PROJECT_PATH 
    python scripts/train_volleyball_stage1.py
    
    # Collective Activity dataset
    cd PROJECT_PATH 
    python scripts/train_collective_stage1.py
  ```

## Acknowledgement

We thank for the part of code of DIN and ST-GCN, whose github repo are [GAP_SRAM code](https://github.com/junwenchen/GAP_SRAM) and [DIN code](https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark). We thank the authors for releasing their code.


## Citation
```
@inproceedings{zhai2023learning,
  title={Learning Group Residual Representation for Group Activity Prediction},
  author={Zhai, Xiaolin and Hu, Zhengxi and Yang, Dingye and Wu, Shichao and Liu, Jingtai},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={300--305},
  year={2023},
  organization={IEEE}
}

```
