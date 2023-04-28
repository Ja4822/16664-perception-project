# 16664-perception-project

## Training
To train, run:
```
python script/train.py --path absolute_path_to_data --logdir log/ --other_params
```
It will (1) load the training data by reading `absolute_path_to_data/labels.csv`; (2) save `config.json`, tensorboard log file, and `model.pt` to `current_path/log/`.

## Evaluation
To evaluate a trained model, run:
```
python script/eval.py --data_path absolute_path_to_data --model_path absolute_path_to_model --logdir log/
```
It will (1) load the evaluation data from `absolute_path_to_data`; (2) load trained model from `absolute_path_to_model/model.pt`; (3) save `pred_labels.csv` to `current_path/log/`.
