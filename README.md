# BlueBoat – Learning-Based Control

## Project overview

This project was developed during an engineering internship at ENSTA Bretagne. Its objective is to develop an autonomous controller for a BlueBoat, allowing it to follow a circular trajectory around a buoy. A vector field controller is used to generate a training dataset, which is then used to train a neural network through supervised learning. Both control approaches can be tested in simulation and deployed on the real BlueBoat.

## Installation

Clone the repository:

```bash
git clone https://github.com/MatBenigno/blueboat_2026.git
cd blueboat_2026
```

Create a Python virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib pandas torch pymavlink
```

## 1. Neural network training and simulation

All scripts related to training and simulation are located in the `blueboat_nn_1` directory.

```bash
cd blueboat_nn_1
mkdir -p data
```

The simulation parameters, desired radius, controller gains and dataset limits can be modified in `config_pool.py`.

### Dataset generation

Generate the training dataset using the vector field controller:

```bash
python3 generate_balanced_dataset.py
```

### Neural network training

Train the neural network using the generated dataset:

```bash
python3 nn_training.py
```

The trained model is saved as `boat_neural_network.pth`. The neural network has a 4-8-8-2 architecture with ReLU activation functions and no biases.

### Weight export

Export the 112 trained weights to a CSV file:

```bash
python3 export_weights.py
```

The weights are saved in `data/neural_network_weights.csv`.

### Simulation and evaluation

Test the trained neural network in simulation:

```bash
python3 nn_simulator_with_weights.py
```

Evaluate the neural network performance:

```bash
python3 evaluate_nn.py
```

The original vector field controller can also be tested:

```bash
python3 simulator_vector_field.py
```

## 2. Real BlueBoat control

All scripts related to communication and control of the real BlueBoat are located in the `blueboat_bassin_2` directory.

From the repository root, copy the exported weights:

```bash
mkdir -p blueboat_bassin_2/data

cp blueboat_nn_1/data/neural_network_weights.csv \
   blueboat_bassin_2/data/

cd blueboat_bassin_2
```

Before running the controller, connect your computer to the BlueBoat network and configure the MAVLink connection, GPS reference point and control parameters in `config_blueboat.py`.

In `main_nn.py`, update the neural network weights path to:

```python
weights = load_neural_network("data/neural_network_weights.csv")
```

Make sure that the GPS, heading, RC channel mapping and emergency stop are working correctly before starting autonomous navigation.

### Vector field controller

```bash
python3 main_vf.py
```

### Neural network controller

```bash
python3 main_nn.py
```

Both controllers use MAVLink to communicate with the BlueBoat and send motor commands through RC Override.

Press `Ctrl+C` to stop the controller.

### Experimental results

The navigation data is automatically recorded in CSV files. To visualize the experimental trajectories, run:

```bash
python3 plot_log_vector_field_real.py
```

or:

```bash
python3 plot_log_neural_network_real.py
```

## Reference

[Carousel Challenge – ENSTA](https://webperso.ensta.fr/Jaulin/carousel.html)
