python entrypoint.py infer \
--model_filepath ./model/id-00000002/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ./model/id-00000002/clean-example-data \
--round_training_dataset_dirpath /mnt/casbat01/r12models/cyber-pdf-dec2022-train/ \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--scale_parameters_filepath ./scale_params.npy
