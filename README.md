# Supplementary Code and Data for "Network Formation and Dynamics among Multi-LLMs"

## Installing Dependencies 

To install the required packages use 

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

### Simulation outputs for the real-world datasets

The simulation outputs for the real-world datasets can be found [here](https://drive.google.com/drive/folders/1pP-zOe4XS--5MArs6Hr4_hUmRhgomzzK?usp=drive_link). To include them in the project, download them and place them in the `outputs/` folder. 


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