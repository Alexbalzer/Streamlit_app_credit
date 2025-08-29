import streamlit as st
import os
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import numpy as np

# =========================================================
# Pfade & Konfiguration
# =========================================================
# Nutze eine Umgebungsvariable ARTIFACTS_DIR, falls gesetzt, sonst ./artifacts
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts")).resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CANDIDATES = [
    "xgb_credit_model.pkl",
    "extre_trees_credit_model.pkl",   # falls du dieses Modell mal speicherst
]

TARGET_COL = "Risk"
CAT_COLS = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]

# =========================================================
# Hilfsfunktionen
# =========================================================
def artifact_path(name: str) -> Path:
    """Erzeuge den absoluten Pfad innerhalb des artifacts-Ordners."""
    return ARTIFACTS_DIR / name

def load_first_existing(paths):
    for p in paths:
        if artifact_path(p).exists():
            return joblib.load(artifact_path(p)), artifact_path(p).name
    return None, None

@st.cache_resource
def load_artifacts():
    # Modell laden
    model, used_model_name = load_first_existing(MODEL_CANDIDATES)
    if model is None:
        st.error(
            "Kein Modell gefunden. Lege z. B. eine 'xgb_credit_model.pkl' in den Ordner "
            f"'{ARTIFACTS_DIR}' oder trainiere und speichere dort."
        )
        st.stop()

    # Feature-Reihenfolge
    feature_names_pkl = artifact_path("feature_names.pkl")
    if feature_names_pkl.exists():
        feature_order = joblib.load(feature_names_pkl)
    else:
        st.error(f"'{feature_names_pkl.name}' nicht gefunden. Bitte dorthin ablegen.")
        st.stop()

    # Encoder
    encoders = {}
    missing = []
    for col in CAT_COLS:
        # Standardname: "<col>_encoder.pkl" (mit evtl. Leerzeichen)
        pkl_name = f"{col}_encoder.pkl"
        p = artifact_path(pkl_name)
        if p.exists():
            encoders[col] = joblib.load(p)
        else:
            # Fallbacks: manchmal nutzt man alternative Suffixe
            # (z. B. *_encoder_credit.pkl). Such diese optional:
            alt = artifact_path(f"{col}_encoder_credit.pkl")
            if alt.exists():
                encoders[col] = joblib.load(alt)
            else:
                missing.append(col)

    if missing:
        st.error(
            "Fehlende Encoder im Ordner 'artifacts': "
            + ", ".join(missing)
            + ". Bitte dort ablegen."
        )
        st.stop()

    return model, used_model_name, encoders, feature_order

def options(col, encoders):
    enc = encoders.get(col)
    return list(getattr(enc, "classes_")) if enc is not None else []

def encode_value(col, value, encoders):
    enc = encoders.get(col)
    if enc is None:
        return value
    # robust gegen unbekannte Kategorien
    if value not in enc.classes_:
        value = enc.classes_[0]
    return enc.transform([value])[0]

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="wide")
st.title("Credit Risk Prediction")
st.caption("Demo für den German Credit Data Datensatz")

with st.spinner("Lade ML-Model und Artefakte…"):
    model, used_model_name, ENCODERS, FEATURE_ORDER = load_artifacts()

if "model_toast_done" not in st.session_state:
    st.toast(f"Modell geladen: {used_model_name} (aus '{ARTIFACTS_DIR.name}')")
    st.session_state["model_toast_done"] = True

with st.sidebar:
    st.header("Eingaben")

    # Defaults
    age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
    sex = st.selectbox("Sex", options=options("Sex", ENCODERS) or ["male", "female"])
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1, step=1)
    housing = st.selectbox("Housing", options=options("Housing", ENCODERS))
    saving_accounts = st.selectbox("Saving accounts", options=options("Saving accounts", ENCODERS))
    checking_account = st.selectbox("Checking account", options=options("Checking account", ENCODERS))
    purpose = st.selectbox("Purpose", options=options("Purpose", ENCODERS))
    credit_amount = st.number_input("Credit amount", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (months)", min_value=1, value=12, step=1)

    st.divider()
    st.subheader("Einstellungen")
    threshold = st.slider("Entscheidungsschwelle ('Bad' ab):", 0.05, 0.95, 0.50, 0.01)

# Modell-konforme Eingabe
row = {
    "Age": age,
    "Sex": encode_value("Sex", sex, ENCODERS),
    "Job": job,
    "Housing": encode_value("Housing", housing, ENCODERS),
    "Saving accounts": encode_value("Saving accounts", saving_accounts, ENCODERS),
    "Checking account": encode_value("Checking account", checking_account, ENCODERS),
    "Credit amount": credit_amount,
    "Duration": duration,
    "Purpose": encode_value("Purpose", purpose, ENCODERS),
}
input_df = pd.DataFrame([row])[FEATURE_ORDER]

tab_pred, tab_whatif, tab_explain, tab_about = st.tabs(
    ["Predictions", "What-if", "Erklärungen", "About"]
)

with tab_pred:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Modell-konformes Input")
        st.dataframe(input_df)
        st.info(f"Aktives Modell: **{used_model_name}** (Ordner: **{ARTIFACTS_DIR.name}**)")

    with c2:
        st.subheader("Vorhersage")
        if st.button("Predict risk"):
            X = input_df
            proba = model.predict_proba(X)[0]
            p_bad = float(proba[1])
            is_bad = p_bad >= threshold
            st.metric("Bad-Risiko", f"{p_bad:.1%}")
            st.progress(min(max(p_bad, 0.0), 1.0))
            if is_bad:
                st.error(f"Einstufung: **BAD** ≥ {threshold:.0%}")
            else:
                st.success(f"Einstufung: **GOOD** < {threshold:.0%}")

            with st.expander("Was bedeutet das?"):
                st.write(
                    """
                    * Das Modell liefert eine Wahrscheinlichkeit für 'bad' (Ausfallrisiko).
                    * Über den Schwellenwert steuerst du, ab wann daraus eine binäre
                      Entscheidung (GOOD/BAD) wird.
                    """
                )

with tab_whatif:
    st.subheader("What-if-Analyse: Wie ändert sich das Risiko, wenn ich ein Feature variiere?")
    feature_to_vary = st.selectbox("Feature wählen", FEATURE_ORDER)
    is_categorical = feature_to_vary in ENCODERS
    base = input_df.iloc[0].copy()

    if is_categorical:
        cats = list(ENCODERS[feature_to_vary].classes_)
        picked_cat = st.selectbox("Wert setzen", cats)
        vary_values_enc = ENCODERS[feature_to_vary].transform(cats)
        min_val = max_val = None
        raw_values = cats
    else:
        current_val = float(base[feature_to_vary])
        c1, c2 = st.columns(2)
        with c1:
            min_default = max(0.0, current_val * 0.5)
            max_default = max(current_val * 1.5, current_val + 1.0)
            min_val, max_val = st.slider(
                "Bereich",
                min_value=0.0,
                max_value=float(max_default * 2),
                value=(float(min_default), float(max_default)),
                step=1.0,
            )
        with c2:
            steps = st.slider("Schritte", 3, 50, 11)
        vary_values_enc = np.linspace(min_val, max_val, int(steps))
        raw_values = [float(v) for v in vary_values_enc]

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("What-if berechnen"):
            st.session_state["whatif_active"] = True
    with b2:
        if st.button("Auto-Update stoppen"):
            st.session_state["whatif_active"] = False

    if st.session_state.get("whatif_active", False):
        if (min_val is not None) and (max_val is not None) and (max_val <= min_val):
            st.warning("Max muss größer als Min sein.")
        else:
            probs = []
            labels = []
            for val_raw, val_enc in zip(raw_values, vary_values_enc):
                row_mut = base.copy()
                row_mut[feature_to_vary] = val_enc
                X_mut = pd.DataFrame([row_mut])[FEATURE_ORDER]
                p_bad = float(model.predict_proba(X_mut)[0][1])
                probs.append(p_bad)
                labels.append(val_raw)

            chart_df = pd.DataFrame({"Wert": labels, "Bad-Risiko": probs})

            fig = px.bar(
                chart_df,
                x="Wert",
                y="Bad-Risiko",
                text="Bad-Risiko",
                color="Bad-Risiko",
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(
                yaxis=dict(title="Risiko"),
                xaxis=dict(title=feature_to_vary),
                title=f"What-if Analyse für {feature_to_vary}",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_about:
    st.write(
        f"""
        **Artefakt-Ordner:** `{ARTIFACTS_DIR}`
        
        Du kannst den Ordner über die Umgebungsvariable **ARTIFACTS_DIR** umstellen:
        ```bash
        set ARTIFACTS_DIR=D:\\pfad\\zu\\artefakten   # Windows (cmd)
        export ARTIFACTS_DIR=/pfad/zu/artefakten    # macOS/Linux
        ```
        """
    )
