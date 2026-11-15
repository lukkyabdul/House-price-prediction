# House Price Prediction

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the FastAPI backend in one terminal:

```powershell
python -m uvicorn src.main:app --reload
```

Start the Streamlit frontend in another terminal:

```powershell
python -m streamlit run src/frontend.py
```

The frontend defaults to `http://127.0.0.1:8000`; change the API Base URL in
the sidebar to connect to a different backend.