Skin Classifier above variable datasets

## Datasets
We use public dataset from hugginface, kaggle and ISIC 2019 Challenge datasets
- [AICamp-2023-Skin-Conditions_Datset](https://huggingface.co/datasets/notable12/AICamp-2023-Skin-Conditions-Dataset)
- [Skin_Conditions_Kaggle](https://www.kaggle.com/datasets/syedalinaqvi/augmented-skin-conditions-image-dataset)
- [ISIC 2019 Challenge dataset](https://challenge.isic-archive.com/data/#2019)
- [MIT-skin-lesions-classification-dataset](https://huggingface.co/datasets/ahmed-ai/skin-lesions-classification-dataset)


## Model
We compare variable models for skin lesion classification
Todo (table)

## Environment:
use `pip install -r requirements.txt` and python version = `3.12.9`


## Usage
We can train & valid and test once using `main.py`
`python main.py --dataset ./data/MIT_skin-lesions-classification-dataset --model convnext -lr 0.00001`

and we can only test using `test.py`
`python test.py --dataset ./data/AICamp-2023-Skin-Conditions_Dataset --model convnext_tiny  --model_save_path 0218_1702_convnext_tiny.pt`

if you don't use wandb, then add arguments --wandb_flag False when you test
`python test.py --dataset ./data/AICamp-2023-Skin-Conditions_Dataset --model convnext_tiny  --model_save_path 0218_1702_convnext_tiny.pt --wandb_flag False`

then make below directories
- results: results of test csv
- save_weights: saved model `.pt` files
- train_val_log: train & val log -> if you use wandb, unnecessary directory


## Grad-CAM
We can visualize for each models final point of features
`grad_cam.ipynb`