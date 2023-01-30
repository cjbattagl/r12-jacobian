import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import log_loss
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    use_feature_reduction_algorithm,
)
# from IPython import embed
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        # TODO: Update skew parameters per round
        # self.model_skew = {
        #     "__all__": metaparameters["infer_cyber_model_skew"],
        # }

        # self.input_features = metaparameters["train_input_features"]
        # self.weight_table_params = {
        #     "random_seed": metaparameters["train_weight_table_random_state"],
        #     "mean": metaparameters["train_weight_table_params_mean"],
        #     "std": metaparameters["train_weight_table_params_std"],
        #     "scaler": metaparameters["train_weight_table_params_scaler"],
        # }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """

        # TODO: This is still just random, not a grid search
        for random_seed in np.random.randint(1000, 9999, 10):
            # self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        check_models_consistency(model_repr_dict)

        # Build model layer map to know how to flatten
        logging.info("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")

        # Flatten models
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened.")


        # logging.info("Feature reduction applied. Creating feature file...")
        X = None
        y = []

        for _ in range(len(flat_models)):
            (model_arch, models) = flat_models.popitem()
            model_index = 0

            logging.info("Parsing %s models...", model_arch)
            for _ in tqdm(range(len(models))):
                model = models.pop(0)
                y.append(model_ground_truth_dict[model_arch][model_index])
                model_index += 1
                # embed()
                model_feats = use_feature_reduction_algorithm(model)
                if X is None:
                    X = model_feats
                    continue

                X = np.vstack((X, model_feats))

        from sklearn.model_selection import train_test_split
        num_trials=4
        roc_sum = 0
        ce_sum = 0
        roc_sum_rf = 0
        ce_sum_rf = 0
        # embed()
        # for trial in range(num_trials):
        #     T_t, T_h, y_t, y_h = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
        #     pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=4, n_estimators=100))
        #     neigh = BaggingClassifier(base_estimator=pipeline, n_estimators=100, random_state=1, n_jobs=16)
        #     neigh.fit(T_t, y_t)
        #     myprob = neigh.predict_proba(T_h)[:,1]
        #     roc_sum += roc_auc_score(y_h, myprob)
        #     ce_sum += metrics.log_loss(y_h, myprob)
        #     # print(neigh.feature_importances_)

        # print('avg roc:', roc_sum / num_trials, 'avg ce:', ce_sum / num_trials)


        logging.info("Training RandomForestRegressor model...")
        # model = RandomForestClassifier()
        # model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=4, n_estimators=400))
        model = BaggingClassifier(base_estimator=pipeline, n_estimators=100, random_state=1, n_jobs=16)
        model.fit(X, y)

        logging.info("Saving RandomForestRegressor model to " + self.model_filepath)
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")

    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(self.scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]

        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()
                # embed()
                pred = torch.argmax(model(feature_vector).detach()).item()

                ground_tuth_filepath = examples_dir_entry.path + ".json"

                with open(ground_tuth_filepath, 'r') as ground_truth_file:
                    ground_truth =  ground_truth_file.readline()

                print("Model: {}, Ground Truth: {}, Prediction: {}".format(examples_dir_entry.name, ground_truth, str(pred)))

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """
        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        with open(self.models_padding_dict_filepath, "rb") as fp:
            models_padding_dict = pickle.load(fp)

        model, model_repr, model_class = load_model(model_filepath)
        model_repr = pad_model(model_repr, model_class, models_padding_dict)
        flat_model = flatten_model(model_repr, model_layer_map[model_class])

        ####
        # self.inference_on_example_data(model, examples_dirpath)

        X = (
            use_feature_reduction_algorithm(flat_model)
        )

        logging.info("Reading classifier from " + self.model_filepath + "...")
        with open(self.model_filepath, "rb") as fp:
            regressor: RandomForestRegressor = pickle.load(fp)


        probability = str(regressor.predict_proba(X.reshape(1,-1))[0][1])

        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
