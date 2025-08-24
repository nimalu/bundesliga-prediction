<img src="https://www.kicktipp.de/assets/kt-DmI-3dQm.svg" width="100" />

# Bundesliga Match Prediction 

This project predicts the results of the current Bundesliga season based on previous seasons. 


The workflow of the project is divided into 6 steps:

1. **Data Acquisition** – Collecting data from previous seasons from [OpenLigaDB](https://www.openligadb.de/)
23. **Data Exploration** – Analyzing the data and identifying important patterns
4. **Baseline** – Evaluate a naive baseline
5. **Poisson Model** – Fit and evaluate a Poisson model
6. **Deployment** – Submitting the predictions to the kicktipp competition


## Installation

1. Clone the repository

```bash
git clone https://github.com/nimalu/bundesliga-tipping.git
cd bundesliga-tipping
```

2. Create environment
```bash
uv sync
```

## License

This project is licensed under the MIT License

## Contact

For any questions or issues, feel free to reach out:  mail@niklaslutze.de