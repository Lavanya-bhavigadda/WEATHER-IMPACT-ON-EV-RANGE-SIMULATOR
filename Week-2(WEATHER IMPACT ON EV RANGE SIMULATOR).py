import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from contextlib import suppress
from io import BytesIO
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

try:
    import speech_recognition as sr
except:
    sr = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
except:
    canvas = None

# ---------------------------------------------------------
# FORCE DARK MODE ALWAYS
# ---------------------------------------------------------
st.set_page_config(page_title="EV Range – Advanced", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0e1117 !important;
    color: #e0e0e0 !important;
}
div, input, textarea, .stTabs, .stTextInput>div>div>input {
    color: #e0e0e0 !important;
}
.stButton>button, .stDownloadButton>button {
    background: #1f2937 !important;
    color: #e0e0e0 !important;
    border: 1px solid #374151 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONSTANTS (using your dataset columns)
# ---------------------------------------------------------
DATA_DEFAULT = "EV_Energy_Consumption_Dataset.csv"

REG_PKL = "weather_week2_reg.pkl"
CLF_PKL = "weather_week2_clf.pkl"

FEATURES = ['Temperature_C', 'Humidity_%', 'Speed_kmh', 'Battery_State_%']
TARGET = 'Distance_Travelled_km'

BINS = [0, 50, 150, np.inf]
LABELS = ['Low', 'Medium', 'High']

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def infer_columns(df):
    """Auto-map similar column names."""
    mapping = {}
    low = {c.lower(): c for c in df.columns}

    def find(*keys):
        for key in keys:
            for lo, orig in low.items():
                if key in lo:
                    return orig
        return None

    guess = {
        'Temperature_C': find("temp"),
        'Humidity_%': find("humid"),
        'Speed_kmh': find("speed"),
        'Battery_State_%': find("batt", "soc"),
        'Distance_Travelled_km': find("distance", "range")
    }

    for new, old in guess.items():
        if old and new not in df.columns:
            mapping[old] = new

    return df.rename(columns=mapping) if mapping else df


def ensure(df):
    needed = FEATURES + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing: {missing}")


def train_models(df):
    df = infer_columns(df).dropna()
    ensure(df)

    X = df[FEATURES]
    y = df[TARGET]

    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=350, random_state=42)
    reg.fit(Xtr, ytr)

    df["Range_Category"] = pd.cut(df[TARGET], bins=BINS, labels=LABELS)
    ycat = df["Range_Category"]

    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X, ycat, test_size=0.2, stratify=ycat, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=350, random_state=42)
    clf.fit(Xtr2, ytr2)

    joblib.dump(reg, REG_PKL)
    joblib.dump(clf, CLF_PKL)
    return reg, clf


def predict_range(reg, clf, t, h, s, b):
    X = pd.DataFrame([[t, h, s, b]], columns=FEATURES)
    rng = float(reg.predict(X)[0])
    cat = clf.predict(X)[0]
    return rng, cat


def make_pdf(text):
    b = BytesIO()
    if canvas is None:
        b.write(text.encode()); b.seek(0); return b

    pkt = BytesIO()
    c = canvas.Canvas(pkt, pagesize=A4)

    width, height = A4
    y = height - 40
    for line in text.splitlines():
        c.drawString(40, y, line)
        y -= 18
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    b.write(pkt.getvalue())
    b.seek(0)
    return b

# ---------------------------------------------------------
# SIDEBAR – DATASET
# ---------------------------------------------------------
st.sidebar.title("Dataset Settings")

upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
path = st.sidebar.text_input("Dataset Path", value=DATA_DEFAULT)

if upload:
    df_upload = pd.read_csv(upload)
    df_upload.to_csv(path, index=False)
    st.sidebar.success("✅ Uploaded & Saved")

force = st.sidebar.checkbox("Force Retrain", value=False)
if force:
    with suppress(Exception): os.remove(REG_PKL)
    with suppress(Exception): os.remove(CLF_PKL)

# Load dataset
if not os.path.exists(path):
    st.error("Dataset not found. Please upload a file.")
    st.stop()

df = pd.read_csv(path)
df = infer_columns(df)

# Train/load models
if os.path.exists(REG_PKL) and os.path.exists(CLF_PKL) and not force:
    reg = joblib.load(REG_PKL)
    clf = joblib.load(CLF_PKL)
else:
    reg, clf = train_models(df)

st.success("✅ Models Loaded Successfully")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_chat, tab_predict, tab_eda, tab_3d, tab_export = st.tabs([
    "💬 Chatbot",
    "🎯 Predictor",
    "📊 EDA",
    "🧊 3D Visuals",
    "⬇️ Export"
])

# ---------------------------------------------------------
# CHATBOT
# ---------------------------------------------------------
FAQ = {
    "temperature": "Temperature affects EV range because batteries lose efficiency in extreme heat or cold.",
    "humidity": "Humidity increases air density, slightly reducing range and forcing AC load.",
    "speed": "Higher speed increases aerodynamic drag → lower EV range.",
    "battery": "Higher battery state usually gives better efficiency and more range.",
    "feature importance": "Feature importance shows how much each feature influences the model.",
    "features": "We use Temperature, Humidity, Speed, and Battery State as inputs.",
    "algorithm": "We use Random Forest for both regression and classification.",
    "target": "The target variable is Distance_Travelled_km.",
    "metrics": "Regression uses R², MAE, RMSE. Classification uses Accuracy.",
    "bins": "Range categories: Low <50 km, Medium 50–150 km, High >150 km.",
    "improve": "Improve by tuning hyperparameters, cleaning data, adding wind/elevation."
}

def answer_faq(msg):
    m = msg.lower()
    for key, val in FAQ.items():
        if key in m:
            return val
    return None

def extract_params(text):
    t = h = s = b = None
    for k, v in re.findall(r"(\w+)\s*=?\s*(-?\d+\.?\d*)", text):
        v = float(v)
        lk = k.lower()
        if "temp" in lk: t = v
        elif "hum" in lk: h = v
        elif "speed" in lk: s = v
        elif "batt" in lk or "soc" in lk: b = v
    return t, h, s, b

with tab_chat:
    st.subheader("Chat with EV Assistant")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        st.chat_message(role).write(msg)

    user = st.chat_input("Ask something...")

    if user:
        st.session_state.chat.append(("user", user))

        ans = answer_faq(user)
        if ans is None:
            t, h, s, b = extract_params(user)
            if None not in (t, h, s, b):
                rng, cat = predict_range(reg, clf, t, h, s, b)
                ans = f"✅ Estimated Range: {rng:.2f} km\n✅ Category: {cat}"
            else:
                ans = "Try: predict temp=25 hum=50 speed=60 batt=80"

        st.session_state.chat.append(("assistant", ans))
        st.rerun()

# ---------------------------------------------------------
# PREDICTOR
# ---------------------------------------------------------
with tab_predict:
    st.subheader("Enter Values for Prediction")

    c1, c2, c3, c4 = st.columns(4)
    with c1: t = st.slider("Temperature (°C)", -30.0, 60.0, 25.0)
    with c2: h = st.slider("Humidity (%)", 0, 100, 50)
    with c3: s = st.slider("Speed (km/h)", 0, 200, 60)
    with c4: b = st.slider("Battery State (%)", 0, 100, 80)

    if st.button("Predict"):
        rng, cat = predict_range(reg, clf, t, h, s, b)
        st.metric("Estimated Range (km)", f"{rng:.2f}")
        st.write(f"Category: **{cat}**")

# ---------------------------------------------------------
# EDA
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Quick EDA")
    st.dataframe(df.head())

    fig, ax = plt.subplots(figsize=(7,4))
    ax.imshow(df[FEATURES+[TARGET]].corr(), cmap="coolwarm")
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(7,3))
    ax2.hist(df[TARGET], bins=30)
    ax2.set_title("Range Distribution")
    st.pyplot(fig2)

# ---------------------------------------------------------
# 3D VISUALS
# ---------------------------------------------------------
with tab_3d:
    st.subheader("3D Visualizations")

    Tmin, Tmax = st.slider("Temperature Range", -20.0, 60.0, (-10.0,40.0))
    Smin, Smax = st.slider("Speed Range", 0.0, 200.0, (0.0,120.0))
    Hfix = st.slider("Fixed Humidity", 0,100,50)
    Bfix = st.slider("Fixed Battery", 0,100,80)

    T = np.linspace(Tmin, Tmax, 30)
    S = np.linspace(Smin, Smax, 30)
    TT, SS = np.meshgrid(T, S)

    Xg = pd.DataFrame({
        'Temperature_C': TT.ravel(),
        'Humidity_%': np.full(TT.size, Hfix),
        'Speed_kmh': SS.ravel(),
        'Battery_State_%': np.full(TT.size, Bfix)
    })
    Z = reg.predict(Xg).reshape(TT.shape)

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(TT, SS, Z, cmap=cm.viridis)
    ax.set_title("3D Surface – Predicted Range")
    ax.set_xlabel("Temp"); ax.set_ylabel("Speed"); ax.set_zlabel("Range")
    st.pyplot(fig)

    st.markdown("### 3D Animated Line")
    frames = st.slider("Frames", 5, 60, 20)
    Tline = np.linspace(Tmin, Tmax, frames)
    Sline = np.linspace(Smin, Smax, frames)
    Xline = pd.DataFrame({
        'Temperature_C': Tline,
        'Humidity_%': np.full(frames,Hfix),
        'Speed_kmh': Sline,
        'Battery_State_%': np.full(frames,Bfix)
    })
    Zline = reg.predict(Xline)

    fig2 = plt.figure(figsize=(7,5))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(Tline, Sline, Zline, lw=2)
    ax2.set_title("3D Line Animation")
    st.pyplot(fig2)

    st.markdown("### 3D Histogram")
    bins = st.slider("Bins", 5,20,10)
    Ts = np.random.uniform(Tmin,Tmax,800)
    Ss = np.random.uniform(Smin,Smax,800)

    Xh = pd.DataFrame({
        'Temperature_C': Ts,
        'Humidity_%': np.full(800,Hfix),
        'Speed_kmh': Ss,
        'Battery_State_%': np.full(800,Bfix)
    })
    Zs = reg.predict(Xh)

    H, xedges, yedges = np.histogram2d(Ts, Ss, bins=bins)
    fig3 = plt.figure(figsize=(7,5))
    ax3 = fig3.add_subplot(111, projection='3d')

    xpos, ypos = np.meshgrid((xedges[:-1]+xedges[1:])/2,
                             (yedges[:-1]+yedges[1:])/2, indexing='ij')
    xpos=xpos.ravel(); ypos=ypos.ravel()
    zpos=np.zeros_like(xpos)
    dx=dy=(xedges[1]-xedges[0])*0.8
    dz=H.ravel()

    ax3.bar3d(xpos,ypos,zpos,dx,dy,dz,shade=True)
    ax3.set_title("3D Histogram")
    st.pyplot(fig3)

# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------
with tab_export:
    st.subheader("Generate PDF")

    tpdf = st.number_input("Temperature", -20.0,60.0,25.0)
    hpdf = st.number_input("Humidity", 0.0,100.0,50.0)
    spdf = st.number_input("Speed", 0.0,200.0,60.0)
    bpdf = st.number_input("Battery", 0.0,100.0,80.0)

    if st.button("Create PDF"):
        rng,cat = predict_range(reg, clf, tpdf,hpdf,spdf,bpdf)

        text = (
            "EV Range Prediction Report\n"
            "----------------------------\n"
            f"Temperature: {tpdf}\n"
            f"Humidity: {hpdf}\n"
            f"Speed: {spdf}\n"
            f"Battery: {bpdf}\n\n"
            f"Estimated Range: {rng:.2f} km\n"
            f"Category: {cat}\n"
        )

        pdf = make_pdf(text)
        st.download_button("Download PDF", data=pdf, file_name="prediction.pdf")
