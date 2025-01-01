# DEELE-Rad: exploiting deep radiomics features in deep learning models using COVID-19 chest X-ray images

This repository contains the code used in **DEELE-Rad (Deep Learning-based Radiomics)** proposal.

**Journal**: [Health Information Science and Systems](https://link.springer.com/journal/13755)

**Authors**: [Márcus V. L. Costa](https://github.com/usmarcv), [Erikson J. de Aguiar](https://github.com/eriksonJAguiar), [Lucas S. Rodrigues](https://github.com/lsrusp), Caetano Traina Jr. and Agma J. M. Traina

**Contents**: [[`Paper`](https://link.springer.com/article/10.1007/s13755-024-00330-6)] [[`Dataset`](https://github.com/usmarcv/deele-rad/tree/main/dataset_script)] [[`Quickstart and Installation`](#quickstart-and-installation)] [[`BibTeX`](#reference)] [[`Contact`](#contact)]

## Quickstart and Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/usmarcv/deele-rad.git
    cd deele-rad
    ```

2. The following instructions should be followed with Python 3.12 to create a Pipenv with all required installed packages. If you do not have Pipenv installed, run the following:
    ```sh
    pip install pipenv
    ```
    
    Activate the environment:
      ```sh
      pipenv shell
      ```
    
    You can install the dependencies libraries based on the `Pipfile` with the following command:
      ```bash
      pipenv sync
      ```

## Running

To run our DEELE-Rad proposal, you can use two approaches:

1. Single model deep learning:
    ```sh
    pipenv run python main.py --model_name VGG15 --num_deep_radiomics 300 --epochs 100
    ```
2. Many models using our `script.sh` file:
    ```sh
    ./script.sh
    ```
    `Note:` you can change the arguments to using another hyperparameters

## Reference

If you use this repository, please cite the following paper:
    
    @inproceedings{Costa2024,
    author={Costa, Márcus V. L. and de Aguiar, Erikson J. and Rodrigues, Lucas S. and Traina, Caetano and Traina, Agma J. M.},
    journal={Health Information Science and Systems (HISS)}, 
    title={{DEELE-R}ad: exploiting deep radiomics features in deep learning models using {COVID-19} chest {X}-ray images}, 
    year={2024},
    volume={13},
    number={},
    pages={517-522},
    doi={10.1007/s13755-024-00330-6},
    url={https://link.springer.com/article/10.1007/s13755-024-00330-6}
    }

## Contact

For more information, you can contact me by writing to [marcusvlc@usp.br](marcusvlc@usp.br) or [LinkedIn](https://www.linkedin.com/in/marcusvlc/).
