import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Residential Energy Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Residential Annual Energy Prediction Tool")
st.write(
    "This application predicts annual residential energy consumption "
    "using a trained XGBoost surrogate model."
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("best_energy_prediction_model_final_features.pkl")

model = load_model()

# ------------------------------------------------------------
# Model feature order
# ------------------------------------------------------------

feature_cols = [
    "apartment_size",
    "floor_position",
    "num_exterior_facades",
    "apartment_exposure_config",
    "building_orientation",
    "wwr",
    "window_glazing",
    "wall_insulation",
    "roof_insulation",
    "ground_floor_f_factor",
    "site_exposure",
    "shading"
]

# ------------------------------------------------------------
# Maps and training ranges
# ------------------------------------------------------------

facade_map = {
    "NW": 2,
    "N": 1,
    "NE": 2,
    "SW": 2,
    "S": 1,
    "SE": 2
}

training_ranges = {
    "wwr": (0.20, 0.45),
    "building_orientation": (0, 135),
    "ground_floor_f_factor": (0.69, 1.29),
    "num_exterior_facades": (1, 2)
}

friendly_names = {
    "apartment_size": "Apartment Size",
    "floor_position": "Floor Position",
    "num_exterior_facades": "Number of Exterior Exposed Façades",
    "apartment_exposure_config": "Apartment Exposure Configuration",
    "building_orientation": "Building Orientation",
    "wwr": "Window-to-Wall Ratio",
    "window_glazing": "Window Glazing Type",
    "wall_insulation": "Wall Insulation Level",
    "roof_insulation": "Roof Insulation Level",
    "ground_floor_f_factor": "Ground Floor F-Factor",
    "site_exposure": "Site Exposure Category",
    "shading": "External Shading Condition"
}

risk_flags = []

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def select_with_other(label, options, key):
    selected = st.selectbox(label, options + ["Other"], key=key)

    if selected == "Other":
        custom_value = st.text_input(
            f"Enter custom value for {label}",
            key=f"{key}_custom"
        )
        if custom_value:
            risk_flags.append(f"{label}: unseen/custom category")
            return custom_value
        return options[0]

    return selected


def numeric_select_with_other(label, options, key, min_val, max_val, default_val):
    options_str = [str(v) for v in options]
    selected = st.selectbox(label, options_str + ["Other"], key=key)

    if selected == "Other":
        custom_value = st.number_input(
            f"Enter custom value for {label}",
            value=float(default_val),
            step=0.01,
            key=f"{key}_custom"
        )

        if custom_value < min_val or custom_value > max_val:
            risk_flags.append(
                f"{label}: outside training range ({min_val}–{max_val})"
            )
        elif custom_value not in options:
            risk_flags.append(
                f"{label}: interpolation within training range"
            )

        return float(custom_value)

    return float(selected)


# ------------------------------------------------------------
# Input panel
# ------------------------------------------------------------

st.subheader("Input Building Characteristics")

col1, col2 = st.columns(2)

with col1:
    apartment_size = select_with_other(
        "Apartment Size",
        ["1room", "2room", "3room"],
        "apartment_size"
    )

    floor_position = select_with_other(
        "Floor Position",
        ["on ground", "middle", "Top"],
        "floor_position"
    )

    apartment_exposure_config = select_with_other(
        "Apartment Exposure Configuration",
        ["NW", "N", "NE", "SW", "S", "SE"],
        "apartment_exposure_config"
    )

    building_orientation = numeric_select_with_other(
        "Building Orientation (degrees)",
        [0, 45, 90, 135],
        "building_orientation",
        0,
        135,
        0
    )

    wwr = numeric_select_with_other(
        "Window-to-Wall Ratio (WWR)",
        [0.20, 0.30, 0.45],
        "wwr",
        0.20,
        0.45,
        0.30
    )

    window_glazing = select_with_other(
        "Window Glazing Type",
        ["triple-windows", "double-windows"],
        "window_glazing"
    )

with col2:
    wall_insulation = select_with_other(
        "Wall Insulation Level",
        ["wall-lowR", "wall-conventional", "wall-upgraded"],
        "wall_insulation"
    )

    roof_insulation = select_with_other(
        "Roof Insulation Level",
        ["roof-conventional", "roof-upgraded"],
        "roof_insulation"
    )

    ground_floor_f_factor = numeric_select_with_other(
        "Ground Floor F-Factor",
        [0.95, 1.29, 0.69],
        "ground_floor_f_factor",
        0.69,
        1.29,
        0.95
    )

    site_exposure = select_with_other(
        "Site Exposure Category",
        ["City", "Country"],
        "site_exposure"
    )

    shading = select_with_other(
        "External Shading Condition",
        ["AlwaysOn", "AlwaysOff"],
        "shading"
    )

# ------------------------------------------------------------
# Derive number of exterior facades
# ------------------------------------------------------------

if apartment_exposure_config in facade_map:
    num_exterior_facades = facade_map[apartment_exposure_config]
else:
    num_exterior_facades = st.number_input(
        "Number of Exterior Exposed Façades",
        min_value=0,
        max_value=4,
        value=2,
        step=1
    )

    if num_exterior_facades < 1 or num_exterior_facades > 2:
        risk_flags.append(
            "Number of Exterior Exposed Façades: outside training range (1–2)"
        )
    else:
        risk_flags.append(
            "Number of Exterior Exposed Façades: manually entered for custom exposure configuration"
        )

# ------------------------------------------------------------
# Build input dataframe
# ------------------------------------------------------------

input_df = pd.DataFrame([{
    "apartment_size": apartment_size,
    "floor_position": floor_position,
    "num_exterior_facades": num_exterior_facades,
    "apartment_exposure_config": apartment_exposure_config,
    "building_orientation": building_orientation,
    "wwr": wwr,
    "window_glazing": window_glazing,
    "wall_insulation": wall_insulation,
    "roof_insulation": roof_insulation,
    "ground_floor_f_factor": ground_floor_f_factor,
    "site_exposure": site_exposure,
    "shading": shading
}], columns=feature_cols)

st.subheader("Selected Building Characteristics")
st.dataframe(input_df, use_container_width=True)

# ------------------------------------------------------------
# Reliability logic
# ------------------------------------------------------------

def get_reliability(flags):
    if len(flags) == 0:
        return (
            "High",
            "All selected inputs exactly match the training design space."
        )

    if any(
        "outside" in flag.lower()
        or "unseen" in flag.lower()
        or "custom" in flag.lower()
        for flag in flags
    ):
        return (
            "Low",
            "At least one input is outside the training domain or represents an unseen/custom category."
        )

    return (
        "Moderate",
        "Some inputs require interpolation but remain within the training range."
    )

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("Predict Annual Energy Consumption"):
    pred_kwh = model.predict(input_df)[0]
    pred_j = pred_kwh * 3_600_000

    reliability, reliability_note = get_reliability(risk_flags)

    st.success("Prediction completed successfully.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            "Annual Energy",
            f"{pred_kwh:,.2f} kWh/year"
        )

    with col_b:
        st.metric(
            "Annual Energy",
            f"{pred_j:,.2e} J/year"
        )

    with col_c:
        st.metric(
            "Reliability",
            reliability
        )

    if reliability == "High":
        st.success(reliability_note)
    elif reliability == "Moderate":
        st.warning(reliability_note)
    else:
        st.error(reliability_note)

    if risk_flags:
        st.subheader("Reliability Notes")
        for flag in risk_flags:
            st.write(f"- {flag}")