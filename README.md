# End-to-End Localization and Ranking for Relative Attributes
Code for the [End-to-End Localization and Ranking for Relative Attributes, ECCV 2016]
Krishna Kumar Singh, Yong Jae Lee
(http://krsingh.cs.ucdavis.edu/krishna_files/papers/relative_attributes/relative_attributes.html)

If you use our work, please cite it:
```bibtex
@inproceedings{krishna-eccv2016,
  title = {End-to-End Localization and Ranking for Relative Attributes},
  author = {Krishna Kumar Singh and Yong Jae Lee},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2016}
}
```  

## Pre-requisites
1. Torch (http://torch.ch/docs/getting-started.html)
2. Torch Libraries `loadcaffe`, `hdf5`, `gnuplot`
3. Download the `stnbhwd-master` from above and install it by executing `luarocks make` inside the folder. This is modified version of orginal Spatial Transformer Netork(STN) code (https://github.com/qassemoquab/stnbhwd). In this version an extra loss is added in `AffineTransformMatrixGenerator.lua` to keep STN within the image boundary.
4. Download `weight-init.lua` (obtained from torch toolkit at https://github.com/e-lab/torch-toolbox/tree/master/Weight-init).
5. Download BVLC reference caffenet (https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) 

## Dataset and Pre-Trained Models
1. Download ECCV_2016 folder from https://drive.google.com/open?id=0B9fXH9R3A3pYSUFpbllLZkZZd0E . It contains training and test data for LFW-10 dataset as well as pre-trained models.
2. Copy the `faces_train` and `faces_test` inside `train_test_data` folder. Training and test data are in `hdf5` format. It contains image pairs (`data` and `datap`) and label indicating whether they have equal attribute strength or `data` has higher strength. The images are mean subtracted and in BGR format.
3. Copy the `models_localization` and `models_combined` in the `learned_model` folder. `models_localization` contains the version of network with just STN with no global image. This model is better for localizing attributes. `models_combined` contained the version of network with both STN and global image. This version is better for ranking. 

## Testing Pre-Trained Models
Models can be tested using `attribute_localization_ranking_testing.lua` code. You can specify only STN (1) and combined model (2) with `modeltype` argument.
Attribute can be specified by `attribute_num` argument. Give value 1  to 10 corresponding to 10 attributes.
1:baldhead, 2:darkhair, 3:eyesopen, 4:goodlooking, 5:masculinelooking, 6:mouthopen, 7:smile, 8:vforehead, 9:v_teeth, 10:young

For example if you want to test combine model (containing both STN and global image) for attribute darkhair
```
th attribute_localization_ranking_testing.lua -attribute_num 2 -modeltype 2
```

For more information use command
```
th attribute_localization_ranking_testing.lua --help
```

## Training Pre-Trained Models
Models can be trained using `attribute_localization_ranking_testing.lua` code. First setup the `alexnet_model_path`, `alexnet_prototxt_path` and `output_dir_path` varialbe. Similar to test code `modeltype` and `attribute_num` can be used to specify type of model and attribute respectively.

For example if you want to train combine model (containing both STN and global image) for attribute smile
```
th attribute_localization_ranking_training.lua -attribute_num 7 -modeltype 2
```

For more information use command
```
th attribute_localization_ranking_training.lua --help
```
NOTE: 

1. For training combine model, STN only model has to be trained first.

2. Scaling is more sensitive than translation. So, if you have issue of convergence during training try to decrease value `scale_ratio` argument.

3. Learned modle will be stored in `output_dir_path`. Code also genrates visualization webpage which shows where STN is localizing over different epochs of the training. 	

## Demo Code
`attribute_demo.lua` shows the localization and ranking results on pair of images stored at `demo_data/input_images/`. Localization results are stored at `demo_data/output_images` and ranking score is printed. `attribute_num` can be used to specify attribute for which demo code will be run.

For example if you want demo for attribute mouth open then run
```
th attribute_demo -attribute_num 6
```

  
