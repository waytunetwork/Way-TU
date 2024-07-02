# WayTU Project

This repository contains the code for the WayTU project, including the setup instructions, dependencies, and a brief overview of the project structure.

## Installation

1. Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Create a virtual Python environment with Python 3.10.12:
    ```bash
    python3 -m venv WayTU-env
    ```

3. Activate the virtual environment:
    ```bash
    source WayTU-env/bin/activate
    ```

4. Install the required robotic package version 0.0.27:
    ```bash
    pip install robotic==0.0.27
    ```

5. Test the installation:
    ```bash
    python3 -c 'from robotic import ry; ry.test.RndScene()'
    ```

6. Install PyTorch version 11.8 and related packages:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

7. Install additional libraries:
    ```bash
    pip install open3d
    pip install -U scikit-learn
    pip install scipy
    ```

## Usage Details

Before running the code, ensure that the paths represented by `...` are updated according to your personal device configuration. All configuration changes can be done in `config.yaml`.

### Collecting Training Samples and Testing the Model

- To collect training samples or test the model, execute `rai_sample.ipynb`.

### Training the Model

- To train the model, execute `train.ipynb`.

## Project Structure

- **Way_TU Model**: Contains model-related classes.
    - `WayTu_Dataset.py`: Contains the custom Torch dataset class.
    - `WayTuModel.py`: Contains the model classes.
    - `train_model.py`: Contains the model training loop.
    - `test_model.py`: Contains two functions: one for testing the model with RAI, and the other for collecting samples.

- **WayTU_Rai**: Contains environment-related classes.

## Additional Information

- Configuration changes can be done using `config.yaml`.
- Ensure that the paths in the configuration file are updated according to your setup.

Special thanks to [PointNet-PyTorch](https://github.com/meder411/PointNet-PyTorch.git) for their PointNet reimplementation.
