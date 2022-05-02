# A Gaussian Process Model for Opponent Prediction in Autonomous Racing
This repository contains the code for the paper:

**<a href="http://arxiv.org/abs/2204.12533">A Gaussian Process Model for Opponent Prediction in Autonomous Racing</a>**
<br>
Finn Lukas Busch, 
Jake Johnson,
Edward L. Zhu, and
Francesco Borrelli
<br>

<!-- Published in ... -->


## Software Requirements
`FORCES PRO` version 4.9+

## Installation instructions
Run `install.sh` to install `barcgp` python package.

## Data Generation
Run `scripts/gen_training_data.py` to generate a series of training samples across different track types. This will generate new FORCES controllers to match those used for data generation.

Parameters:
- `policy_name`: Determines policy that the target vehicle will use in training
- `track_types`: Track types that will be generated
- `total_runs`: Number of sample runs

## Running Simulations
Run `scripts/run_sim.py` to simulate a head-to-head race with a predictor modeling the prediction of the target vehicle.

Parameters:
- `predictor_class`: Predictor to use (type of [`ConstantVelocityPredictor`, `ConstantAngularVelocityPredictor`, `GPPredictor`, `NLMPCPredictor`])
- `policy_name`: Determines policy that the target vehicle will use in simulation
- `use_GPU`: Whether to use GPU for inference when using `GPPredictor`
- `M`: Number of samples to generate from GP predictor
- `T`: Max length in seconds of experiment
- `gen_scenario`: Controls whether to generate new scenario or use saved pkl
- `use_predictions_from_module`: Set to true to use predictions generated from `predictor_class`, otherwise use true predictions from MPCC
