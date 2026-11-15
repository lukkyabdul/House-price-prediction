import pandas as pd
import requests
import streamlit as st

DEFAULT_API_BASE_URL = "https://house-price-prediction-2yae.onrender.com/"
REQUEST_TIMEOUT_SECONDS = 60
HOUSE_COLUMNS = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
INTEGER_COLUMNS = {"bedrooms", "bathrooms", "age_years"}
SAMPLE_HOUSES = [
    {"area_sqft": 1850.0, "bedrooms": 3, "bathrooms": 2, "age_years": 5},
    {"area_sqft": 1200.0, "bedrooms": 2, "bathrooms": 2, "age_years": 10},
    {"area_sqft": 2400.0, "bedrooms": 4, "bathrooms": 3, "age_years": 2},
]
THEMES = {
    "Light": {
        "background": "#ffffff",
        "surface": "#FAF8F8",
        "text": "#121212",
        "muted": "#536273",
        "border": "#d5dce3",
    },
    "Dark": {
        "background": "#0e1117",
        "surface": "#1b2029",
        "text": "#852222",
        "muted": "#dde5ed",
        "border": "#3a4552",
    },
}


def _validation_messages(response: requests.Response) -> list[str]:
    try:
        payload = response.json()
    except ValueError:
        return [f"API returned HTTP {response.status_code} with an invalid JSON response."]

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list):
        messages = []
        for error in detail:
            if not isinstance(error, dict):
                messages.append(str(error))
                continue

            location = error.get("loc", [])
            row = next(
                (part + 1 for part in location if isinstance(part, int)),
                None,
            )
            field = next(
                (
                    str(part).replace("_", " ")
                    for part in reversed(location)
                    if isinstance(part, str)
                    and part not in {"body", "houses"}
                    and not part.isdigit()
                ),
                "Input",
            )
            prefix = f"Row {row}, " if row is not None else ""
            message = str(error.get("msg", "Invalid input"))
            message = message.removeprefix("Value error, ").removeprefix("Value error: ")
            messages.append(f"{prefix}{field}: {message}")
        return messages or ["The API rejected the request."]

    if isinstance(detail, str):
        return [detail]
    return [f"API returned HTTP {response.status_code}: {payload}"]


def _show_api_errors(response: requests.Response) -> None:
    for message in _validation_messages(response):
        st.error(message)


def _request_failed_message() -> str:
    return (
        "Could not reach the FastAPI backend. Start it in a terminal with "
        "`python -m uvicorn src.main:app --reload`, then try again."
    )


def _apply_theme(theme_name: str) -> None:
    colors = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
            :root {{
                color-scheme: {"dark" if theme_name == "Dark" else "light"};
                --app-background: {colors["background"]};
                --app-surface: {colors["surface"]};
                --app-text: {colors["text"]};
                --app-muted: {colors["muted"]};
                --app-border: {colors["border"]};
            }}
            html, body, [data-testid="stAppViewContainer"],
            [data-testid="stHeader"], [data-testid="stToolbar"] {{
                background-color: var(--app-background);
                color: var(--app-text);
            }}
            [data-testid="stSidebar"] > div {{
                background-color: var(--app-surface);
            }}
            [data-testid="stAppViewContainer"] *,
            [data-testid="stSidebar"] * {{
                color: var(--app-text);
            }}
            [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
            [data-testid="stWidgetLabel"] p, [data-testid="stHelp"] {{
                color: var(--app-muted);
            }}
            input, textarea, [data-baseweb="select"] > div,
            [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
                background-color: var(--app-surface);
                border-color: var(--app-border);
            }}
            [data-testid="stMetric"] {{
                background-color: var(--app-surface);
                border: 1px solid var(--app-border);
                border-radius: 0.75rem;
                padding: 1rem;
            }}
            [data-testid="stTabs"] button {{
                color: var(--app-text);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _health_status(api_base_url: str) -> None:
    st.sidebar.subheader("API Health Status")
    if not api_base_url:
        st.sidebar.warning("Enter an API base URL to check backend health.")
        return

    try:
        response = requests.get(
            f"{api_base_url}/health",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException:
        st.sidebar.error(
            "FastAPI server is unreachable. Start it with "
            "`python -m uvicorn src.main:app --reload`."
        )
        return

    if response.status_code != 200:
        st.sidebar.error(f"Health check failed (HTTP {response.status_code}).")
        return

    try:
        health = response.json()
    except ValueError:
        st.sidebar.error("Health check returned an invalid JSON response.")
        return

    if isinstance(health, dict) and health.get("model_loaded") is True:
        st.sidebar.success("Healthy — model loaded")
    elif isinstance(health, dict):
        st.sidebar.warning(
            f"API status: {health.get('status', 'unknown')} — model not loaded"
        )
    else:
        st.sidebar.error("Health check returned an unexpected response.")


def _house_payloads(frame: pd.DataFrame) -> list[dict[str, int | float]]:
    houses = []
    for row_number, row in enumerate(frame[HOUSE_COLUMNS].itertuples(index=False), 1):
        house = {}
        for column, value in zip(HOUSE_COLUMNS, row):
            if pd.isna(value):
                raise ValueError(f"Row {row_number}: {column.replace('_', ' ')} is required.")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Row {row_number}: {column.replace('_', ' ')} must be numeric."
                ) from None

            if column in INTEGER_COLUMNS:
                if not number.is_integer():
                    raise ValueError(
                        f"Row {row_number}: {column.replace('_', ' ')} must be a whole number."
                    )
                house[column] = int(number)
            else:
                house[column] = number
        houses.append(house)
    return houses


def main() -> None:
    st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
    st.title("House Price Prediction — FastAPI + Streamlit Client (Team 09)")

    theme_name = st.sidebar.selectbox("Theme", options=list(THEMES), index=0)
    _apply_theme(theme_name)
    st.sidebar.header("Backend connection")
    api_base_url = st.sidebar.text_input(
        "API Base URL",
        value=DEFAULT_API_BASE_URL,
        help="The FastAPI server address, without a trailing slash.",
    ).strip().rstrip("/")
    _health_status(api_base_url)

    single_tab, batch_tab = st.tabs(
        ["Single Property Prediction", "Batch Property Prediction"]
    )

    with single_tab:
        st.subheader("Estimate a property's price")
        if st.button("Test Invalid Input (422 Demo)", key="invalid_input_demo"):
            st.session_state["single_area_sqft"] = 300.0
            st.session_state["single_bedrooms"] = 6
            st.session_state["single_bathrooms"] = 2
            st.session_state["single_age_years"] = 5
            st.info("Demo preset loaded: 300 sq.ft with 6 bedrooms. Submit to see the API reject it.")

        with st.form("single_prediction_form"):
            area_sqft = st.number_input(
                "Area (sq.ft)",
                min_value=250.01,
                max_value=20000.0,
                step=50.0,
                key="single_area_sqft",
            )
            bedrooms = st.slider(
                "Bedrooms", min_value=1, max_value=10, key="single_bedrooms"
            )
            bathrooms = st.slider(
                "Bathrooms", min_value=1, max_value=10, key="single_bathrooms"
            )
            age_years = st.slider(
                "Property age (years)", min_value=0, max_value=100, key="single_age_years"
            )
            submitted = st.form_submit_button("Predict price", type="primary")

        if submitted:
            if not api_base_url:
                st.error("Enter an API base URL in the sidebar.")
            else:
                payload = {
                    "area_sqft": area_sqft,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "age_years": age_years,
                }
                try:
                    response = requests.post(
                        f"{api_base_url}/predict",
                        json=payload,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                except requests.exceptions.RequestException:
                    st.error(_request_failed_message())
                else:
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            predicted_price = result["predicted_price_lakhs"]
                        except (ValueError, KeyError, TypeError):
                            st.error("The prediction API returned an invalid response.")
                        else:
                            st.metric(
                                "Predicted price",
                                f"₹{predicted_price:,.2f} Lakhs",
                            )
                            st.caption(
                                f"Model: {result.get('model_version', 'Unknown')} · "
                                f"API latency: "
                                f"{response.headers.get('X-Process-Time-Ms', 'unavailable')} ms"
                            )
                    else:
                        _show_api_errors(response)

    with batch_tab:
        st.subheader("Predict prices for multiple properties")
        uploaded_file = st.file_uploader(
            "Optional: upload a CSV with area_sqft, bedrooms, bathrooms, and age_years",
            type=["csv"],
            key="batch_csv_upload",
        )
        if st.button("Load CSV into table", disabled=uploaded_file is None):
            try:
                csv_frame = pd.read_csv(uploaded_file)
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
                st.error(f"Could not read the CSV: {error}")
            else:
                missing_columns = [column for column in HOUSE_COLUMNS if column not in csv_frame]
                if missing_columns:
                    st.error(
                        "CSV is missing required columns: " + ", ".join(missing_columns)
                    )
                else:
                    st.session_state["batch_houses_data"] = csv_frame[HOUSE_COLUMNS]
                    if "batch_houses_editor" in st.session_state:
                        del st.session_state["batch_houses_editor"]
                    st.success(f"Loaded {len(csv_frame)} CSV row(s) into the table.")

        if "batch_houses_data" not in st.session_state:
            st.session_state["batch_houses_data"] = pd.DataFrame(SAMPLE_HOUSES)
        edited_houses = st.data_editor(
            st.session_state["batch_houses_data"],
            key="batch_houses_editor",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "area_sqft": st.column_config.NumberColumn(
                    "Area (sq.ft)", format="%.2f", step=50.0
                ),
                "bedrooms": st.column_config.NumberColumn("Bedrooms", step=1),
                "bathrooms": st.column_config.NumberColumn("Bathrooms", step=1),
                "age_years": st.column_config.NumberColumn("Age (years)", step=1),
            },
        )

        if st.button("Predict Batch", type="primary"):
            if not api_base_url:
                st.error("Enter an API base URL in the sidebar.")
            elif edited_houses.empty:
                st.error("Add at least one house to the table.")
            else:
                try:
                    houses = _house_payloads(edited_houses)
                except ValueError as error:
                    st.error(str(error))
                else:
                    try:
                        response = requests.post(
                            f"{api_base_url}/predict-batch",
                            json={"houses": houses},
                            timeout=REQUEST_TIMEOUT_SECONDS,
                        )
                    except requests.exceptions.RequestException:
                        st.error(_request_failed_message())
                    else:
                        if response.status_code == 200:
                            try:
                                result = response.json()
                                predictions = result["predictions_lakhs"]
                                count = result["count"]
                            except (ValueError, KeyError, TypeError):
                                st.error("The batch prediction API returned an invalid response.")
                            else:
                                if len(predictions) != len(edited_houses):
                                    st.error(
                                        "The API returned a different number of predictions "
                                        "than submitted houses."
                                    )
                                else:
                                    results = edited_houses[HOUSE_COLUMNS].copy()
                                    results["predicted_price_lakhs"] = predictions
                                    st.metric("Properties predicted", count)
                                    st.caption(
                                        f"API latency: "
                                        f"{response.headers.get('X-Process-Time-Ms', 'unavailable')} ms · "
                                        f"Currency: {result.get('currency', 'INR')}"
                                    )
                                    st.dataframe(
                                        results,
                                        hide_index=True,
                                        width="stretch",
                                    )
                        else:
                            _show_api_errors(response)


if __name__ == "__main__":
    main()
