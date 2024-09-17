<img src="https://www.kicktipp.de/assets/kt-DmI-3dQm.svg" width="100" />

# Bundesliga Match Prediction 

This project predicts the results of the current Bundesliga season based on various statistics from the previous seasons. Using features such as goals scored and points achieved in the last season, the goal is to forecast match outcomes using machine learning models.



The workflow of the project is divided into 6 steps:

1. **Data Acquisition** – Collecting data from previous seasons from [OpenLigaDB](https://www.openligadb.de/)
2. **Feature Engineering** – Generating and processing features for model input, using _pandas_
3. **Data Exploration** – Analyzing the data and identifying important patterns, using _matplotlib_
4. **Model Building** – Training machine learning models, using _scikit-learn_, _xgboost_, ...
5. **Evaluation** – Assessing model performance based on Kicktipp scoring
6. **Deployment** – Submitting the predictions to the kicktipp competition


## Installation

1. Clone the repository

```bash
git clone https://github.com/nimalu/bundesliga-tipping.git
cd bundesliga-tipping
```

2. Set up a Conda environment
```bash
conda env create -f environment.yaml
conda activate bundesliga-prediction
```

## License

This project is licensed under the MIT License

## Contact

For any questions or issues, feel free to reach out:  mail@niklaslutze.de