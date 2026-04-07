This project is a research project which trys to predict commercial flight departure and arrival delays. 
The goal is to build useful models that can be stored and used by other software to give consumers more information.
the general workflow is: src/fetch_prune/fetch_all_flights start end
prepare_dataset.py dep_config.json
features_dep.py dep_config.json
features_arr.py arr_config.json
train_dep_bins_ordinal_catboost.py dep_train_config.json
train_arr_bins_ordinal_catboost.py arr_train_config.json
