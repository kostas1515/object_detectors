<h2>Instructions for YOLOv3</h2>
This file contains instructions on how to train YOLOv3 on COCO and LVIS datasets.

Please install the following libraries:
- torch 
- torchvision
- numpy 
- pandas
- sklearn
- scipy
- skimage
- cv2
- imgaug
- apex
- hydra-core
- logging
- tensorboard
- pycocotools
- lvis

This repo uses hydra to orchestrate, maintain and develop experiments. For more info on hydra visit their [mainpage](https://hydra.cc/docs/intro)
After downloading the dataset
Link it's location by modifying the file named "hydra/dataset\<dataset\>.yaml".
One can tune multiple hyperparameters by using hydra's syntact in the command line or one can change the yaml deafult values.
The file hydra/config.yaml contains the default configs/values.

To train the default configuration one can use:
<code>
    python main.py
</code>

To train the bayesian optimised configuration one can use the command from batch_files/sample.txt.


