🧪 Overview
Coordination‑ML is a machine‑learning project designed to analyze, model, and predict properties of coordination compounds using quantitative chemical descriptors.
The system integrates:

atomic descriptors

ligand descriptors

chelation features

crystal field theory

stability constants

ML classification & regression models

The project is fully modular, scalable, and structured for production use with FastAPI + Streamlit.

🧱 Project Structure
Code
coordination-ml/
│
├── api/                 # FastAPI backend
├── configs/             # YAML configuration files
├── data/                # Raw, processed, external datasets
│   ├── raw/
│   ├── processed/
│   └── external/
├── features/            # Feature engineering pipeline
├── models/              # Training, evaluation, saved models
├── notebooks/           # Jupyter experiments
├── src/                 # Core chemistry + ML logic
├── tests/               # Unit tests
└── ui/                  # Streamlit interface
📊 Features
1. Feature Engineering
The pipeline computes:

electronegativity

ionization energy

atomic radius

HSAB hardness

ligand denticity

donor atom aggregation

charge density proxies

ligand field splitting (Δ₀)

stability constants (log Kf)

geometry labels (octahedral, tetrahedral, square planar)

2. Machine Learning Models
Models include:

Geometry Classifier (RandomForest)

Ligand Strength Predictor

Stability Regression Models

All models are saved under:

Code
models/saved/
3. API Layer (FastAPI)
Endpoints for:

geometry prediction

ligand strength prediction

stability prediction

4. Streamlit UI
Interactive interface for:

exploring ligand features

predicting geometry

visualizing chemical descriptors

inspecting model outputs

📂 Data Files
Required Raw Files
Place these inside data/raw/:

atomic_properties.csv

ligands_raw.csv

complexes_raw.csv

Processed Files
Generated automatically:

feature_matrix.csv

feature_matrix.parquet

⚙️ Installation
1. Clone the repository
Code
git clone https://github.com/jagdevsinghdosanjh/coordination-ml.git
cd coordination-ml
2. Install dependencies
Code
pip install -r requirements.txt
🚀 Usage
1. Build Feature Matrix
Code
python features/build_feature_matrix.py
2. Train Geometry Classifier
Code
python models/train/train_geometry_classifier.py
3. Run FastAPI Server
Code
uvicorn api.main:app --reload
4. Launch Streamlit UI
Code
streamlit run ui/app.py
🧬 Scientific Basis
The project uses quantitative descriptors from:

Crystal Field Theory

HSAB Theory

Spectrochemical Series

Donor atom electronegativity & ionization energy

Ligand denticity & chelation strength

Stability constants (Kf)

These descriptors are transformed into ML‑ready features for predictive modeling.

🧪 Example Predictions
Predict whether a complex is octahedral, tetrahedral, or square planar

Estimate ligand strength from Δ₀

Predict stability using log(Kf)

Visualize donor atom contributions

🧱 Tech Stack
Python 3.10+

Pandas, NumPy, Scikit‑Learn

FastAPI

Streamlit

PyYAML

PyArrow

🧑‍💻 Author
Jagdev Singh Dosanjh  
Government High School, Chananke
Amritsar, Punjab, India
Ed‑Tech Founder & Architect of Modular PCM Systems

📜 License
MIT License (or your preferred license)