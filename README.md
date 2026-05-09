## Graphene Nanopore Membrane Predictor (Streamlit)

Web tool to predict:
- **Salt Rejection (%)**
- **Water Flux (molecule/ns)** (internally modeled via **log10(flux+1)**, then converted back)

Using 7 inputs from the dataset:
- `Geometry` (categorical)
- `Pore Area (Å²)` (numeric)
- `Applied pressure  (MPa)` (numeric)
- `Feed Concentration  (ppm)` (numeric)
- `Temperature  (°C)` (numeric)
- `Pore Chemistry (Functionalization)` (categorical)
- `Porosity (%)` (numeric)

### Project structure
- `models/`: trained `.pkl` pipelines + metadata (min/max ranges and categories)
- `utils/`: shared helpers (validation, plotting, export, PDF, chatbot stub)
- `app.py`: Streamlit app (to be added next)

### Local run
Create an environment (recommended) then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Train models (optional)
The repository already contains trained models in `models/`.
If you want to retrain from the Excel file:

```bash
python train_salt_rejection_model.py
python train_log_flux_model.py
```

### Streamlit Cloud deployment
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app from the GitHub repo.
3. Set the main file to `app.py`.
4. Add secrets (later) for the chatbot if needed.

