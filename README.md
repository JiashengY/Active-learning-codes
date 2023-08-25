<a name="readme-top"></a>

# Active-learning-codes
This repository is linked to the research article J. Yang, A. Stroh, S. Lee, S. Bagheri, B. Frohnapfel and P. Forooghi. (2023) Prediction of equivalent sand-grain size and identification of drag-relevant scales of roughness - a data driven approach <em>JFM</em>.

[![PS_PDF_NeuralNetwork](0_assets/NN.png)](https://arxiv.org/abs/2304.08958)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
            <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#Folders">Folders</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
The present project contains python scripts range from [reading roughness statistics](#rgh_category)
,  [Bayesian optimization of NN architecture](#Bayesian_optimization) and [training of ensemble neural network](#enn_training) based on given architecture.



<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="30" height="30" title="Python"/>  &nbsp; &nbsp; v 3.8

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started
The present repository contains jupyter notbooks scripts.

### Prerequisites

Python 3.8 along with following Python packages are required for running the code

* Numpy
  ```bash
    pip3 install numpy==1.21.5
  ```
* Scipy
  ```bash
    pip3 install scipy==1.7.3
  ```

* Pandas
  ```bash
    pip3 install pandas==1.4.1
  ```

* Matplotlib
  ```bash
    pip3 install matplotlib==3.6.2
  ```

* TensorFlow
  ```bash
    pip3 install tensorflow==2.10.0
  ```

* scikit-learn
  ```bash
    pip3 install -U scikit-learn==1.0.2
  ```
* scikit-optimize
  ```bash
    pip3 install scikit-optimize==0.9.0
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

Clone the repository
   ```sh
   git clone https://github.com/JiashengY/Active-learning-codes.git
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
### rgh Python class
**rgh** is a Python class that provides a convenient and efficient way to perform a variety of operations related to roughness analysis.
You can move the rgh_class.py file from this repository and place it in your project directory. Import **rgh** class:
```python 
import rgh_class as rgh
```
Create an instance of the rgh class:
```python 
surface=rgh.rgh(x,z,y) # x,z are streamwise, spanwise coordinate, y is 2-D roughness map
```

### Methods
The **rgh** class provides the following methods:
* **print_stat()** : Generate a dictionary containing roughness statistical parameters
* **show_surface(representation="2D")** : Plot surface geometry.
  * **representation** : String, shoing the representation of roughness geometry either in "2D" or "3D".
* **plot_PDF(n_bins=10,Normalization=True)** : Present roughness height PDF, where:
  * **n_bins**: integer, number of bins. 
  * **Normalization** : boolean, if the plot is noramalized to density.
* **plot_PS(Normalization=True,circular_average=False,moving_average=False)**: Present roughness height PS, where:
  * **Normalization** : boolean, if the PS is noramalized with root-mean-square roughness height. 
  * **azimuthal_average=False**: azimuthally averaged PS around origin of spectral space i.e. (qx,qz)=(0,0), to acquire 1-d PS. 
  * **moving_average=False**: Once ayimuthal average is done, moving average over 1-d PS can be carried out on demand. 
  * **n_iter**: number of moving averaging iterations.
* **FFT_filter(lmax,lmin)**: Spectral filtering of the surface, **lmax**, **lmin** are the desired largest and smallest roughness wavelengths, respectively.
* **get_model_input(lmax=2,lmin=0.04,azimuthal_average=False,moving_average=False,n_iter=3)**: Generate input vector for the ML model, where:
  * **lmax**: largest incorporated wavelength
  * **lmin**: smallest incorporated wavelength
  * **azimuthal_average=False**: azimuthally averaged PS around origin of spectral space i.e. (qx,qz)=(0,0), to acquire 1-d PS. 
  * **moving_average=False**: Once ayimuthal average is done, moving average over 1-d PS can be carried out on demand. 
  * **n_iter**: number of moving averaging iterations.

### Roughness prediction
**predict(surface_input,n_models=50)** is the function to predict given inputs with the ensemble neural network.
* **surface_input** is the input vector for the current model, this can be obtained by **get_model_input** method in **rgh** class. For instance:

```python
prediction=rgh.predict(surface.get_model_input())
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Folders -->
## Folders

* #### rgh_class 
This folder contains roughness class **rgh**, **rgh** is a Python class that provides a convenient and efficient way to perform a variety of operations related to roughness analysis.
* #### Bayesian_optimization
This folder contains BO routine for optimizing NN archeticture for the current task setup. Output file <strong>Hyperparameters_BO.csv</strong>
* #### enn_training
This folder contains ENN training script based on  user specified NN architecture documented in <strong>Hyperparameters.csv</strong>, either generated from BO process or user-defined.


<!-- ciation-->
## Citation

```
@misc{yang2023prediction,
      title={Prediction of equivalent sand-grain size and identification of drag-relevant scales of roughness -- a data driven approach}, 
      author={Jiasheng Yang and Alexander Stroh and Sangseung Lee and Shervin Bagheri and Bettina Frohnapfel and Pourya Forooghi},
      year={2023},
      eprint={2304.08958},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn}
}
```

<!-- contact -->
## Contact
[Pourya Forooghi](https://pure.au.dk/portal/en/persons/pourya-forooghi(22c2cfbf-081f-4494-b545-45ef29ae5d0f).html) @ Aarhus University

[Alexander Stroh](https://www.istm.kit.edu/558_522.php) @ KIT

[Jiasheng Yang](https://www.istm.kit.edu/558_1459.php) @ KIT
<!-- license -->
## License

This repository is released under the [MIT license](https://choosealicense.com/licenses/mit/)



