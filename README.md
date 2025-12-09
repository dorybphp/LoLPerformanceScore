# LoLPerformanceScore - Contextual Favorability Model 
EE P 596 Final Project Autumn 2025

This project builds a context-aware deep learning model that evaluates how historically favorable a player–champion–team configuration is within professional League of Legends.  
Using match data from 2019–2024, the model predicts win probability and produces a continuous favorability score derived from learned embeddings and contextual features.

The goal is *not* to estimate intrinsic skill, but to quantify **how effective a specific configuration has historically been** based on the model’s understanding of player tendencies, champion synergies, and team contexts.


## Repository Structure
├── README.md <br>
├── requirements.txt <br>
├── src/ <br>
│   ├── main.py      # Entry point of the program <br>
│   ├── utils.py     # Any helper functions <br>
│   ├── model.py     # Model definition <br>
├── checkpoints/ <br>
├── demo/            # Full original .ipynb <br>
└── results/ <br>         


## Project Overview
### Motivation
Professional League of Legends is highly contextual:  
player impact depends on role, champion, team identity, region strength, and draft strategy.

Traditional metrics (KDA, DPM, gold diff) fail to capture:

- role expectations  
- synergy with champion picks  
- team style  
- region/tournament strength  
- per-player historical tendencies  

### Solution
This model uses:
- Embeddings for player, champion, team, role, league, tournament  
- Normalized per-player performance stats (role-aware normalization)  
- Team aggregate features  
- A deep MLP with BatchNorm + Dropout  
- Two model heads (win-probability + latent performance signal)

Then, for each configuration:
1. The model classifies a win/loss while simultaneously a producing continuous score that reflects, what the model thinks is, how impactful a player was to a winning outcome of a match.
2. Historical probabilities are aggregated.
3. The aggregated values are scaled to produce a **0–100 favorability score**.

This score answers:  
**"How historically favorable is this player–champion–team configuration?"**


## Dataset

Source: [League of Legends Esports Player Game Data (2019-2024)](https://ieee-dataport.org/documents/league-legends-esports-player-game-data-2019-2024)
- ~370,000 per-player samples originally  
- Filtered to retain players with ≥ 50 games, yielding ~325,000 samples  
- ~1950 unique professional players  
- All numerical stats normalized per role (removes role bias)  
- Categorical features encoded for embeddings
- Numeric team aggregates calculated
- Train/val/test split is time-aware

Raw data is not included in this repository. Due to this dataset requiring a subscribtion to access, only the two neccessary files are contained in the folder **that requires a @uw.edu email to access**. Access to this folder will be removed by the end of the year (2025).
Dataset link: [Drive Folder](https://drive.google.com/drive/folders/139NUY4-yaVzY6BYi0-lVJm2CEh4dElai?usp=sharing)
**! If you have trouble accessing this please notify me immediately.**


## Installation
Follow these steps to install dependencies and run the project.

### 1. Create a virtual environment
Linux/macOS:
```
python -m venv .venv
source .venv/bin/activate
```
Windows: 
```
python -m venv .venv
./.venv/Scripts/Activate.ps1
```

### 2. Upgrade pip
```
pip install --upgrade pip wheel setuptools
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Install pytorch
CPU only
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
GPU (CUDA 11.8 example):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Run the full training pipeline
python src/main.py


## Expected Output
After running `main.py`, users should expect:
- Console logs showing epoch‑by‑epoch training and validation loss/accuracy.
- A final printed evaluation summary on the test set.
- A saved checkpoint at:
```
checkpoints/best_model.pt
```
- Metrics and plots saved in:
```
results/
```


## Pre-trained Model
If you want to skip training, download a pre-trained version of the model [here](https://drive.google.com/drive/folders/1f3aaZvVktkwH0SOvueDE8kwuRvFmbQ8Q?usp=sharing).
Place the file at:
```
checkpoints/model.pt
```
Then run inference or demo scripts normally.


## Demo
Running the demo notebook is the easiest way to reproduce the results.

### Option 1 - Full Project Pipeline
Download the dataset files to the same directory as the notebook and run the entire notebook.

### Option 2 - Evaluation of Existing Pre-trained Model
Download the pre-trained model and run the notebook starting from the "Evaluation" subsection.


## Acknowledgments
This project uses or references the following external resources:
- [League of Legends Esports Player Game Data (2019-2024)](https://ieee-dataport.org/documents/league-legends-esports-player-game-data-2019-2024)
    > Maxime De Bois, Flora Parmentier, Raphaël Puget, Matthew Tanti, Jordan Peltier, "League of Legends Esports Player Game Data (2019-2024)", IEEE Dataport, January 16, 2025, doi:10.21227/5evv-jk25 
- PyTorch for model training
- Pandas, NumPy, scikit‑learn for data handling and preprocessing
