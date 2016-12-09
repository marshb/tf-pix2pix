# tf-pix2pix

tensorflow implementation of [pix2pix](https://github.com/phillipi/pix2pix)

(*still in construction*)

## Setup
```shell
bash ./datasets/download_dataset.sh facades
```

## Train
```shell
python train.py --batch_size=10
```

## Results 
facade --> groundtruth --> generation
![](samples/sample_8000_0.jpg)
![](samples/sample_8000_2.jpg)
![](samples/sample_8000_6.jpg)
![](samples/sample_8000_7.jpg)
![](samples/sample_8000_8.jpg)

credit to 

1. https://github.com/phillipi/pix2pix
2. https://github.com/yenchenlin/pix2pix-tensorflow
