# FPAR-Project-MLDL (todo!)

This directory contains the models that implement First Person Action Recognition using a Resnet34 + CAM and a convLSTM module as a spatial attention network, paired with  a temporal network, again a Resnet34, that takes optical flows as an input to
extract temporal features.
Much of this work is based on the paper: "Swathikiran Sudhakaran and Oswald Lanz. Attention is all we need: Nailing down object-centric attention for egocentric activity recognition. In British Machine Vision Conference, 2018" as also the code.

Models :
- resnetMod.py : contains the implementation of the resnet34
- MyConvLSTMCell.py : implementation of a convLSTM cell
-	objectAttentionModelConvLSTM.py : expands the 2 implementation of resnet and convLSTM in the previous files 
                                    introducing the CAM mechanism paired with the convLSTM module, with a downstream classifier
- flow_resnet.py : implements resnet34 to extract temporal features from optical flow data 
- twoStreaModel.py : paires together the spatial and temporal networks

Transformations:
- spatial_transforms.py : implementation of several of pytorch's transformation, optimized to work with both rgb and
                          optical flow frames

Dataset:
- gtea_dataset.py : to ease the access to the GTEA61 dataset in every possible way. Contains one class to get the rgb frames,
                    one for the optical flow framses, and one that is a wrapper of the other 2, to get them both

Training:
- train_pipeline.ipynb : jupyter notebook with all the training steps.
                         Stage 1) training of the convLSTM module and its downstream classifier
                         Stage 2) train the convLSTM, the classifier and the CAM layer altogether
                         Stage 3) separate training of the temporal network, flow_resnet
                         Stage 4) Joint fine-tuning of the 2 networks
                          
