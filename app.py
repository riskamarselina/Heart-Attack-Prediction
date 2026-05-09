import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Risiko Penyakit Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .result-positive {
        background-color: #fdecea;
        border-left: 6px solid #e74c3c;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-negative {
        background-color: #eafaf1;
        border-left: 6px solid #27ae60;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .var-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-left: 4px solid #3498db;
    }
    .var-card.danger  { border-left-color: #e74c3c; background-color: #fff5f5; }
    .var-card.warning { border-left-color: #f39c12; background-color: #fffbf0; }
    .var-card.normal  { border-left-color: #27ae60; background-color: #f0fff4; }
    .food-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    h1, h2, h3 { color: #2c3e50; }
    .stProgress > div > div > div > div { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
DATASET_PATH = "heart_attack_prediction_indonesia.csv"

CATEGORICAL_COLS = [
    'gender', 'region', 'income_level', 'smoking_status',
    'alcohol_consumption', 'physical_activity', 'dietary_habits',
    'air_pollution_exposure', 'stress_level', 'EKG_results'
]

FEATURE_COLS = [
    'age', 'gender', 'region', 'income_level', 'hypertension', 'diabetes',
    'cholesterol_level', 'obesity', 'waist_circumference', 'family_history',
    'smoking_status', 'alcohol_consumption', 'physical_activity', 'dietary_habits',
    'air_pollution_exposure', 'stress_level', 'sleep_hours',
    'blood_pressure_systolic', 'blood_pressure_diastolic', 'fasting_blood_sugar',
    'cholesterol_hdl', 'cholesterol_ldl', 'triglycerides', 'EKG_results',
    'previous_heart_disease', 'medication_usage'
]

# ─── Train Model (cached — hanya dijalankan SEKALI saat startup) ──────────────
@st.cache_resource(show_spinner=False)
def train_model():
    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURE_COLS].copy()
    y = df['heart_attack'].copy()

    # Handle missing values
    for col in X.select_dtypes(include='number').columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Label encoding
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        max_features='sqrt',
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model, scaler, label_encoders

# Tampilkan spinner hanya saat training pertama kali
if "model_ready" not in st.session_state:
    with st.spinner("⏳ Memuat sistem... (hanya sekali saat pertama dibuka)"):
        model, scaler, label_encoders = train_model()
    st.session_state["model_ready"] = True
else:
    model, scaler, label_encoders = train_model()

# ─── Helper: encode input user ────────────────────────────────────────────────
def encode_input(input_dict):
    df_in = pd.DataFrame([input_dict])
    for col in CATEGORICAL_COLS:
        le = label_encoders[col]
        val = str(df_in[col].iloc[0])
        if val in le.classes_:
            df_in[col] = le.transform([val])
        else:
            df_in[col] = 0
    return df_in[FEATURE_COLS]

# ─── Helper: interpretasi variabel ───────────────────────────────────────────
def interpret_bp(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "normal", "✅ Normal — tekanan darah ideal"
    elif systolic < 130 and diastolic < 80:
        return "warning", "⚠️ Meningkat — perlu dipantau"
    elif systolic < 140 or diastolic < 90:
        return "warning", "⚠️ Hipertensi Tahap 1 (130–139/80–89 mmHg)"
    elif systolic < 180 or diastolic < 120:
        return "danger", "❗ Hipertensi Tahap 2 (≥140/≥90 mmHg) — konsultasi dokter"
    else:
        return "danger", "🚨 Krisis Hipertensi (≥180/≥120 mmHg) — darurat medis!"

def interpret_waist(waist, gender):
    limit = 90 if gender == "Male" else 80
    if waist <= limit:
        return "normal", f"✅ Normal (≤{limit} cm untuk {'Pria' if gender=='Male' else 'Wanita'})"
    elif waist <= limit + 10:
        return "warning", f"⚠️ Sedikit Berlebih — risiko metabolik meningkat"
    else:
        return "danger", f"❗ Obesitas Sentral (>{limit+10} cm) — risiko kardiovaskular tinggi"

def interpret_glucose(fbs):
    if fbs < 100:
        return "normal", "✅ Normal (< 100 mg/dL)"
    elif fbs < 126:
        return "warning", "⚠️ Pre-Diabetes (100–125 mg/dL) — perhatikan pola makan"
    else:
        return "danger", "❗ Indikasi Diabetes (≥126 mg/dL) — konsultasi dokter"

def interpret_hdl(hdl, gender):
    low = 40 if gender == "Male" else 50
    if hdl >= 60:
        return "normal", "✅ Optimal (≥60 mg/dL) — melindungi jantung"
    elif hdl >= low:
        return "warning", f"⚠️ Cukup ({low}–59 mg/dL) — bisa ditingkatkan"
    else:
        return "danger", f"❗ Rendah (<{low} mg/dL) — risiko jantung meningkat"

def interpret_ldl(ldl):
    if ldl < 100:
        return "normal", "✅ Optimal (< 100 mg/dL)"
    elif ldl < 130:
        return "normal", "✅ Mendekati Optimal (100–129 mg/dL)"
    elif ldl < 160:
        return "warning", "⚠️ Batas Tinggi (130–159 mg/dL) — perhatikan diet"
    elif ldl < 190:
        return "danger", "❗ Tinggi (160–189 mg/dL) — konsultasi dokter"
    else:
        return "danger", "❗ Sangat Tinggi (≥190 mg/dL) — risiko sangat tinggi"

def interpret_triglycerides(tg):
    if tg < 150:
        return "normal", "✅ Normal (< 150 mg/dL)"
    elif tg < 200:
        return "warning", "⚠️ Batas Tinggi (150–199 mg/dL)"
    elif tg < 500:
        return "danger", "❗ Tinggi (200–499 mg/dL) — perlu penanganan"
    else:
        return "danger", "❗ Sangat Tinggi (≥500 mg/dL) — darurat medis"

def interpret_sleep(hours):
    if 7 <= hours <= 9:
        return "normal", "✅ Ideal (7–9 jam)"
    elif 6 <= hours < 7 or 9 < hours <= 10:
        return "warning", "⚠️ Kurang ideal — usahakan 7–9 jam"
    else:
        return "danger", "❗ Kurang tidur ekstrem — berdampak pada jantung"

def interpret_age(age):
    if age < 45:
        return "normal", "✅ Relatif muda — risiko dasar rendah"
    elif age < 55:
        return "warning", "⚠️ Usia menengah — perlu pemantauan rutin"
    else:
        return "danger", "❗ Usia lanjut — risiko kardiovaskular meningkat signifikan"

def get_food_recommendations(has_disease, high_bp, high_glucose, high_ldl, high_tg, low_hdl, obese):
    foods_good, foods_avoid = [], []

    if has_disease or high_ldl or low_hdl:
        foods_good += [
            ("🐟 Ikan Berlemak", "Salmon, tuna, sarden — kaya omega-3 untuk jantung"),
            ("🥑 Alpukat", "Lemak sehat yang menurunkan LDL"),
            ("🫐 Beri-berian", "Blueberry, stroberi — antioksidan tinggi"),
            ("🥦 Sayur Hijau", "Brokoli, bayam — serat & folat untuk jantung"),
        ]
        foods_avoid += [
            ("🥩 Daging Merah Berlemak", "Lemak jenuh meningkatkan kolesterol LDL"),
            ("🍟 Gorengan", "Trans fat berbahaya untuk pembuluh darah"),
        ]

    if high_bp:
        foods_good += [
            ("🍌 Pisang", "Kalium tinggi — membantu menurunkan tekanan darah"),
            ("🧄 Bawang Putih", "Allicin terbukti menurunkan tekanan darah"),
            ("🍠 Ubi Jalar", "Kalium & magnesium untuk kesehatan pembuluh darah"),
        ]
        foods_avoid += [
            ("🧂 Garam Berlebih", "Natrium meningkatkan tekanan darah"),
            ("🥫 Makanan Kalengan", "Sodium tersembunyi yang berbahaya"),
        ]

    if high_glucose or high_tg:
        foods_good += [
            ("🌾 Oatmeal", "Serat larut menstabilkan gula darah"),
            ("🥗 Salad Sayur", "Rendah kalori & glikemik — ideal untuk diabetes"),
            ("🫘 Kacang-kacangan", "Protein nabati, serat, rendah glikemik"),
        ]
        foods_avoid += [
            ("🧃 Minuman Manis", "Gula tambahan meningkatkan trigliserida & gula darah"),
            ("🍞 Roti/Nasi Putih", "Indeks glikemik tinggi — ganti dengan versi gandum"),
        ]

    if obese:
        foods_good += [
            ("🥒 Sayuran Rendah Kalori", "Timun, selada, seledri — mengenyangkan tanpa kalori"),
            ("🍵 Teh Hijau", "Meningkatkan metabolisme & mengandung antioksidan"),
        ]
        foods_avoid += [
            ("🍕 Fast Food", "Kalori tinggi, lemak jenuh & sodium berlebih"),
            ("🍰 Dessert Manis", "Kalori kosong yang memperburuk obesitas"),
        ]

    if not foods_good:
        foods_good = [
            ("🥗 Sayur & Buah Segar", "Serat, vitamin, mineral untuk jantung sehat"),
            ("🐟 Ikan", "Protein sehat dengan omega-3"),
            ("🌾 Biji-bijian Utuh", "Sumber karbohidrat kompleks terbaik"),
        ]
    if not foods_avoid:
        foods_avoid = [
            ("🍟 Gorengan", "Lemak trans tidak baik untuk jantung"),
            ("🧂 Makanan Sangat Asin", "Meningkatkan tekanan darah"),
        ]

    # Deduplicate
    foods_good = list(dict(foods_good).items())
    foods_avoid = list(dict(foods_avoid).items())
    return foods_good, foods_avoid

# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🫀 Sistem Deteksi Risiko Penyakit Jantung</h1>
    <p style="font-size:1.1rem; margin:0;">Isi data kesehatan Anda di bawah ini untuk mengetahui risiko serangan jantung</p>
</div>
""", unsafe_allow_html=True)

st.info("ℹ️ Aplikasi ini hanya bersifat **informatif** dan bukan pengganti diagnosis dokter. Konsultasikan hasil ini dengan tenaga medis profesional.")

# ─── Input Form ──────────────────────────────────────────────────────────────
with st.form("health_form"):
    st.subheader("📋 Data Diri & Gaya Hidup")
    col1, col2, col3 = st.columns(3)

    with col1:
        age            = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=45)
        gender         = st.selectbox("Jenis Kelamin", ["Male", "Female"],
                                      format_func=lambda x: "Pria" if x == "Male" else "Wanita")
        region         = st.selectbox("Wilayah Tempat Tinggal", ["Urban", "Rural"],
                                      format_func=lambda x: "Perkotaan" if x == "Urban" else "Pedesaan")
        income_level   = st.selectbox("Tingkat Penghasilan", ["Low", "Middle", "High"],
                                      format_func=lambda x: {"Low":"Rendah","Middle":"Menengah","High":"Tinggi"}[x])
        sleep_hours    = st.slider("Jam Tidur per Hari", 1.0, 14.0, 7.0, 0.5)

    with col2:
        smoking_status      = st.selectbox("Status Merokok", ["Never", "Past", "Current"],
                                           format_func=lambda x: {"Never":"Tidak Pernah","Past":"Mantan Perokok","Current":"Masih Merokok"}[x])
        alcohol_consumption = st.selectbox("Konsumsi Alkohol", ["Moderate", "High"],
                                           format_func=lambda x: {"Moderate":"Sedang","High":"Tinggi"}[x])
        physical_activity   = st.selectbox("Aktivitas Fisik", ["Low", "Moderate", "High"],
                                           format_func=lambda x: {"Low":"Rendah","Moderate":"Sedang","High":"Tinggi"}[x])
        dietary_habits      = st.selectbox("Pola Makan", ["Healthy", "Unhealthy"],
                                           format_func=lambda x: "Sehat" if x == "Healthy" else "Tidak Sehat")
        stress_level        = st.selectbox("Tingkat Stres", ["Low", "Moderate", "High"],
                                           format_func=lambda x: {"Low":"Rendah","Moderate":"Sedang","High":"Tinggi"}[x])

    with col3:
        air_pollution_exposure = st.selectbox("Paparan Polusi Udara", ["Low", "Moderate", "High"],
                                              format_func=lambda x: {"Low":"Rendah","Moderate":"Sedang","High":"Tinggi"}[x])
        family_history         = st.selectbox("Riwayat Penyakit Jantung Keluarga", [0, 1],
                                              format_func=lambda x: "Tidak" if x == 0 else "Ya")
        previous_heart_disease = st.selectbox("Riwayat Penyakit Jantung Sebelumnya", [0, 1],
                                              format_func=lambda x: "Tidak" if x == 0 else "Ya")
        medication_usage       = st.selectbox("Sedang Mengonsumsi Obat Jantung", [0, 1],
                                              format_func=lambda x: "Tidak" if x == 0 else "Ya")
        ekg_results            = st.selectbox("Hasil EKG", ["Normal", "Abnormal"])

    st.markdown("---")
    st.subheader("🩺 Data Klinis (Dari Pemeriksaan Kesehatan)")
    col4, col5, col6 = st.columns(3)

    with col4:
        bp_systolic       = st.number_input("Tekanan Darah Sistolik (mmHg)", 60, 250, 120)
        bp_diastolic      = st.number_input("Tekanan Darah Diastolik (mmHg)", 40, 160, 80)
        fbs               = st.number_input("Gula Darah Puasa (mg/dL)", 50, 500, 90)
        cholesterol_level = st.number_input("Kolesterol Total (mg/dL)", 100, 600, 200)

    with col5:
        cholesterol_hdl = st.number_input("Kolesterol HDL — Baik (mg/dL)", 10, 150, 50)
        cholesterol_ldl = st.number_input("Kolesterol LDL — Jahat (mg/dL)", 30, 400, 100)
        triglycerides   = st.number_input("Trigliserida (mg/dL)", 30, 1000, 150)

    with col6:
        waist        = st.number_input("Lingkar Pinggang (cm)", 40, 200, 80)
        hypertension = st.selectbox("Didiagnosis Hipertensi", [0, 1],
                                    format_func=lambda x: "Tidak" if x == 0 else "Ya")
        diabetes     = st.selectbox("Didiagnosis Diabetes", [0, 1],
                                    format_func=lambda x: "Tidak" if x == 0 else "Ya")
        obesity      = st.selectbox("Didiagnosis Obesitas", [0, 1],
                                    format_func=lambda x: "Tidak" if x == 0 else "Ya")

    submitted = st.form_submit_button("🔍 Analisis Risiko Saya", use_container_width=True)

# ─── Hasil Analisis ───────────────────────────────────────────────────────────
if submitted:
    input_data = {
        'age': age, 'gender': gender, 'region': region, 'income_level': income_level,
        'hypertension': hypertension, 'diabetes': diabetes,
        'cholesterol_level': cholesterol_level, 'obesity': obesity,
        'waist_circumference': waist, 'family_history': family_history,
        'smoking_status': smoking_status, 'alcohol_consumption': alcohol_consumption,
        'physical_activity': physical_activity, 'dietary_habits': dietary_habits,
        'air_pollution_exposure': air_pollution_exposure, 'stress_level': stress_level,
        'sleep_hours': sleep_hours, 'blood_pressure_systolic': bp_systolic,
        'blood_pressure_diastolic': bp_diastolic, 'fasting_blood_sugar': fbs,
        'cholesterol_hdl': cholesterol_hdl, 'cholesterol_ldl': cholesterol_ldl,
        'triglycerides': triglycerides, 'EKG_results': ekg_results,
        'previous_heart_disease': previous_heart_disease, 'medication_usage': medication_usage,
    }

    X_input    = encode_input(input_data)
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    st.markdown("---")
    st.header("📊 Hasil Analisis")

    # ── Diagnosis Utama ───────────────────────────────────────────────────────
    col_r1, col_r2 = st.columns([1.4, 1])
    with col_r1:
        if prediction == 1:
            st.markdown("""
            <div class="result-positive">
                <h2 style="color:#e74c3c;margin:0">❗ BERISIKO SERANGAN JANTUNG</h2>
                <p style="margin:0.5rem 0 0">Berdasarkan data yang Anda masukkan, sistem mendeteksi <b>risiko tinggi</b> penyakit jantung. Segera konsultasikan dengan dokter.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-negative">
                <h2 style="color:#27ae60;margin:0">✅ RISIKO RENDAH SERANGAN JANTUNG</h2>
                <p style="margin:0.5rem 0 0">Berdasarkan data Anda, risiko serangan jantung <b>relatif rendah</b>. Tetap jaga gaya hidup sehat!</p>
            </div>""", unsafe_allow_html=True)

    with col_r2:
        risk_label = ("🔴 Sangat Tinggi" if probability > 0.75 else
                      "🟠 Tinggi"        if probability > 0.5  else
                      "🟡 Sedang"        if probability > 0.25 else "🟢 Rendah")
        st.metric("Probabilitas Risiko", f"{probability * 100:.1f}%", delta=risk_label)
        st.progress(float(probability))

    # ── Interpretasi Variabel ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔎 Keterangan Tiap Variabel Kesehatan Anda")

    gender_label = "Pria" if gender == "Male" else "Wanita"

    vars_list = [
        ("Usia",               *interpret_age(age),
         f"**{age} tahun** — " + interpret_age(age)[1]),
        ("Tekanan Darah",      *interpret_bp(bp_systolic, bp_diastolic),
         f"**{bp_systolic}/{bp_diastolic} mmHg** — " + interpret_bp(bp_systolic, bp_diastolic)[1]),
        ("Gula Darah Puasa",   *interpret_glucose(fbs),
         f"**{fbs} mg/dL** — " + interpret_glucose(fbs)[1]),
        ("Kolesterol HDL",     *interpret_hdl(cholesterol_hdl, gender),
         f"**{cholesterol_hdl} mg/dL** — " + interpret_hdl(cholesterol_hdl, gender)[1]),
        ("Kolesterol LDL",     *interpret_ldl(cholesterol_ldl),
         f"**{cholesterol_ldl} mg/dL** — " + interpret_ldl(cholesterol_ldl)[1]),
        ("Trigliserida",       *interpret_triglycerides(triglycerides),
         f"**{triglycerides} mg/dL** — " + interpret_triglycerides(triglycerides)[1]),
        ("Lingkar Pinggang",   *interpret_waist(waist, gender),
         f"**{waist} cm** — " + interpret_waist(waist, gender)[1]),
        ("Jam Tidur",          *interpret_sleep(sleep_hours),
         f"**{sleep_hours} jam/hari** — " + interpret_sleep(sleep_hours)[1]),
        ("Hipertensi",
         "danger" if hypertension else "normal",
         "❗ Terdiagnosis — faktor risiko utama penyakit jantung" if hypertension else "✅ Tidak terdiagnosis"),
        ("Diabetes",
         "danger" if diabetes else "normal",
         "❗ Terdiagnosis — meningkatkan risiko kardiovaskular 2–4×" if diabetes else "✅ Tidak terdiagnosis"),
        ("Obesitas",
         "danger" if obesity else "normal",
         "❗ Terdiagnosis — beban jantung meningkat" if obesity else "✅ Tidak terdiagnosis"),
        ("Riwayat Keluarga",
         "warning" if family_history else "normal",
         "⚠️ Ada — faktor genetik, gaya hidup sangat memengaruhi risiko" if family_history else "✅ Tidak ada"),
        ("Status Merokok",
         "danger" if smoking_status == "Current" else ("warning" if smoking_status == "Past" else "normal"),
         {"Current":"❗ Masih Merokok — risiko 2–4× lebih tinggi",
          "Past":   "⚠️ Mantan Perokok — risiko menurun tapi masih ada",
          "Never":  "✅ Tidak Merokok — baik untuk jantung"}[smoking_status]),
        ("Pola Makan",
         "danger" if dietary_habits == "Unhealthy" else "normal",
         "❗ Tidak Sehat — perlu perubahan segera" if dietary_habits == "Unhealthy" else "✅ Sehat — pertahankan!"),
        ("Aktivitas Fisik",
         "danger" if physical_activity == "Low" else ("warning" if physical_activity == "Moderate" else "normal"),
         {"Low":     "❗ Rendah — gaya hidup sedentary meningkatkan risiko jantung",
          "Moderate":"⚠️ Sedang — tingkatkan ke 150 menit/minggu",
          "High":    "✅ Tinggi — sangat baik untuk jantung"}[physical_activity]),
        ("Tingkat Stres",
         "danger" if stress_level == "High" else ("warning" if stress_level == "Moderate" else "normal"),
         {"High":    "❗ Tinggi — stres kronis memicu inflamasi & gangguan jantung",
          "Moderate":"⚠️ Sedang — kelola dengan olahraga atau meditasi",
          "Low":     "✅ Rendah — stres terkontrol dengan baik"}[stress_level]),
        ("Hasil EKG",
         "danger" if ekg_results == "Abnormal" else "normal",
         "❗ Abnormal — indikasi gangguan irama/struktur jantung, perlu evaluasi dokter"
         if ekg_results == "Abnormal" else "✅ Normal — tidak ada kelainan terdeteksi"),
    ]

    col_v1, col_v2 = st.columns(2)
    for i, item in enumerate(vars_list):
        label = item[0]; status = item[1]; desc = item[2] if len(item) == 3 else item[3]
        with (col_v1 if i % 2 == 0 else col_v2):
            st.markdown(f"""
            <div class="var-card {status}">
                <b>{label}</b><br>
                <span style="font-size:0.9rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

    # ── Rekomendasi Makanan ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🍽️ Rekomendasi Makanan untuk Kondisi Anda")

    high_bp      = interpret_bp(bp_systolic, bp_diastolic)[0] in ["warning", "danger"]
    high_glucose = interpret_glucose(fbs)[0]                   in ["warning", "danger"]
    high_ldl     = interpret_ldl(cholesterol_ldl)[0]           in ["warning", "danger"]
    high_tg      = interpret_triglycerides(triglycerides)[0]   in ["warning", "danger"]
    low_hdl_flag = interpret_hdl(cholesterol_hdl, gender)[0]   in ["warning", "danger"]

    foods_good, foods_avoid = get_food_recommendations(
        prediction == 1, high_bp, high_glucose, high_ldl, high_tg, low_hdl_flag, obesity == 1
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("### 🟢 Makanan yang Dianjurkan")
        for name, desc in foods_good:
            st.markdown(f"""
            <div class="food-card">
                <b>{name}</b><br>
                <span style="color:#555;font-size:0.88rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with col_f2:
        st.markdown("### 🔴 Makanan yang Perlu Dihindari")
        for name, desc in foods_avoid:
            st.markdown(f"""
            <div class="food-card" style="border-left:3px solid #e74c3c;">
                <b>{name}</b><br>
                <span style="color:#555;font-size:0.88rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.warning("⚕️ **Disclaimer:** Hasil prediksi ini berdasarkan model machine learning dan **bukan diagnosis medis resmi**. Selalu konsultasikan kondisi kesehatan Anda dengan dokter atau tenaga medis yang berwenang.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ Tentang Aplikasi")
    st.markdown("""
Aplikasi ini menggunakan model **Machine Learning (Random Forest)** yang dilatih  
pada dataset *Heart Attack Prediction Indonesia* dengan **158.355 data**.

**Variabel yang dianalisis:**
- Data demografi & gaya hidup  
- Data klinis (tekanan darah, kolesterol, gula darah)  
- Riwayat kesehatan & kondisi medis

**Output:**
- ✅/❗ Prediksi risiko serangan jantung + probabilitas
- 🔎 Keterangan status tiap variabel
- 🍽️ Rekomendasi makanan sesuai kondisi
    """)
    st.markdown("---")
    st.markdown("### 📞 Darurat Medis")
    st.error("Jika mengalami nyeri dada mendadak, segera hubungi:\n\n**☎️ 119 (Ambulans Nasional)**")
