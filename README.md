# Supplementary Code and Data for "Network Formation and Dynamics among Multi-LLMs"

## Installing Dependencies 

The installation on a normal desktop computer should take a few minutes. To install the required packages use 

```bash
pip install -r requirements.txt
```

## Configuring the LLM API

Our experiments use GPT-3.5 and GPT-4 provided by OpenAI's API. To configure your environment please create a file named `params.json` structured as follows

```json
{
    "OPENAI_API_KEY" : "YOUR_API_KEY",
    "OPENAI_ORG" : "YOUR_ORG_KEY"
}
```

## Running the Simulations

The simulation files are located in `principle_X.ipynb`, where

 * `principle_1.ipynb` corresponds to the simulations regarding preferential attachment
 * `principle_2.ipynb` corresponds to the simulations regarding triadic closure
 * `principle_3.ipynb` corresponds to the experiments regarding homophily and their community structure
 * `principle_5.ipynb` corresponds to the experiments regardign small-world properties

Also, `combined_model.ipynb` corresponds to the experiments concerning the real-world datasets (located in `datasets/`).
  
The output figures are saved in `figures/`. The output tables are located in `tables/`. 

The (cached) outputs of the simulations are located in `outputs/`. If you wish to rerun the experiments, erase the contents of this directory. 

### Simulation outputs for the real-world datasets and human baseline data

The simulation outputs for the real-world datasets can be found [here](https://drive.google.com/drive/folders/1nalOHz7fYA7EFegZ1wKuwz56BMuZB1ex?usp=drive_link). 

To include the pre-run simulation outputs in the project, download them and place them in the `outputs/` folder. 
To include the human baseline data in the project, download them and place them in the `user_study_data_public` directory.

### Running the experiments on your own data

In order to run the software on your own data, you need to create a dataloader method in the `dataloader.py` file, similar to `load_facebook100` and change the dataloader in `combined_model.ipynb`. The dataloader function should return a dictionary `networks` which has the labels of (potentially) multiple networks as keys, and the corresponding `nx.Graph` objects as values. Each `nx.graph` object can have features, which should be set using the `nx..set_node_attributes(G, feat_dict, 'features')` command where `feat_dict` is the dictionary of features (see also [here](https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html) for more information). 

### Expected runtimes 

The experiments in `principle_1.ipynb`, `principle_2.ipynb`, `principle_3.ipynb` and `principle_5.ipynb` should take at most an hour each to run using the `gpt-3.5-turbo` model on a normal laptop computer. 

The experiments in `combined_model.ipynb` take several hours (~4-5 hours) each to run using the `gpt-4-1106-preview` model for the datasets in question on a normal laptop computer.

### Test system specifications

The experiments have been run at a MacBook M2 Pro with an Apple M2 Max chip, 32 GB of RAM, running macOS 13.0. The version of python used is Python 3.10.9. 

## Citation

If you use this code, please cite our work as follows

```bibtex
@article{papachristou2024network,
  title={Network Formation and Dynamics Among Multi-LLMs},
  author={Papachristou, Marios and Yuan, Yuan},
  journal={arXiv preprint arXiv:2402.10659},
  year={2024}
}
```