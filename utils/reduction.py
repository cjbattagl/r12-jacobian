import importlib

import numpy as np
from tqdm import tqdm
# from IPython import embed


# def feature_reduction(model, weight_table, max_features):
#     outputs = {}
#     tf = max_features / len(model)
#     sm = sum([l.shape[0] for l in model.values()])
#     for (layer, weights) in model.items():
#         wt_i = np.round(weights.shape[0] / sm * 100).astype(np.int32)
#         out_f = int(weight_table[wt_i] * tf)
#         if layer == list(model.keys())[-1]:
#             out_f = max_features - sum(outputs.values())
#         assert out_f > 0
#         outputs[layer] = out_f
#     return outputs


# def init_feature_reduction(output_feats):
#     fr_algo = "sklearn.decomposition.FastICA"
#     fr_algo_mod = ".".join(fr_algo.split(".")[:-1])
#     fr_algo_class = fr_algo.split(".")[-1]
#     mod = importlib.import_module(fr_algo_mod)
#     fr_class = getattr(mod, fr_algo_class)
#     return fr_class(n_components=output_feats)


# def init_weight_table(random_seed, mean, std, scaler):
#     rnd = np.random.RandomState(seed=random_seed)
#     return np.sort(rnd.normal(mean, std, 100)) * scaler


# def fit_feature_reduction_algorithm(model_dict, weight_table_params, input_features):
#     layer_transform = {}
#     weight_table = init_weight_table(**weight_table_params)

#     for (model_arch, models) in model_dict.items():
#         layers_output = feature_reduction(models[0], weight_table, input_features)
#         layer_transform[model_arch] = {}
#         for (layers, output) in tqdm(layers_output.items()):
#             layer_transform[model_arch][layers] = init_feature_reduction(output)
#             s = np.stack([model[layers] for model in models])
#             layer_transform[model_arch][layers].fit(s)

#     return layer_transform


def use_feature_reduction_algorithm(model):
    mylist = list(model.items())
    layer1 = np.asarray(mylist[-1][1]).squeeze()
    layer2 = np.asarray(mylist[-2][1]).squeeze()
    layer2len = layer2.shape[0]

    out_model = np.hstack((layer2[0:1], layer2[layer2len-1], layer1[0]))
    return out_model
