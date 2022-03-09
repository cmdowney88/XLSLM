# Crosslingual Segmental Language Model

This repository contains the code from
 _Multilingual unsupervised sequence segmentation transfers to extremely low-resource languages_ (2021,
 C.M. Downey, Shannon Drizin, Levon Haroutunian, and Shivin Thukral). The code
 here is a modified version of the repository from [the original MSLM paper](https://github.com/cmdowney88/SegmentalLMs).
 The _mslm_ package can be used to train and use Segmental Language Models.

In this repository, we additionally make available our preparation of the
AmericasNLP 2021 multilingual dataset (see `Data/AmericasNLP`) and the target
K'iche' data (`Data/GlobalClassroom`).

## Paper Results

The results from the accompanying paper can be found in the `Output` directory.
 `*.csv` files include statistics from the training run, `*.out` contain the
  model output for the entire corpus, `*.score` contain the segmentation scores
  of the model output.

The results from the October 2021 pre-print (which we will refer to as
 Experiment Set A) are reproducible on commit `2b89575`. We will consider this the
 official commit of the October 2021 pre-print.
 
The results from the second version of the paper, to appear in the [theme track of
 the ACL 2022 main conference](https://www.2022.aclweb.org/callpapers), were
 produced on commit `820c40e`.
 
## Usage

The top-level scripts for training and experimentation can be found in
 `RunScripts`. Almost all functionality is run through the `__main__.py` script
 in the `mslm` package, which can either train or evaluate/use a model. The
 PyTorch modules for building SLMs can be found in `mslm.segmental_lm`, modules
 for the span-masking Transformer are in `mslm.segmental_transformer`, and
 modules for sequence lattice-based computations are in `mslm.lattice`. The main
 script takes in a configuration object to set most parameters for model
 training and use (see `mslm.mslm_config`). For information on the arguments to
 the main script:
     
    python -m mslm --help

### Environment setup
    pip install -r requirements.txt

This code requires Python >= 3.6

### Training
    ./RunScripts/run_mslm.sh <MODEL/CONFIG_NAME> <TRAIN_FILE> <VALIDATION_FILE>
or 

    python -m mslm --input_file <TRAIN_FILE> \
        --model_path <MODEL_PATH> \
        --mode train \
        --config_file <CONFIG_FILE> \
        --dev_file <DEV_FILE> \
        [--preexisting]

### Evaluation
    ./RunScripts/eval_mslm.sh <INPUT_FILE> <MODEL_PATH> <CONFIG_NAME> <TRAINING_WORDS>

Where `<TRAINING_WORDS>` is a text file containing all of the words from the 
training set
