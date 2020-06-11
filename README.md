This repository contains the code that we used for our submission 'Efficient Poverty Mapping from High Resolution Remote Sensing Images'.

sh run.sh will run the training code
sh run_test.sh will run the testing code

We have provided the checkpoints but unforntunately we can not provide the dataset used in the experiments due to extremely large size of the dataset.

'Uganda2012Consumption_cluster.csv' contains the latitude/longitude and poverty scores for all the villages. We used this to pull satellite images from both Digital Globe Satellites (High-Res) and Sentinel-2 (Low-Res).