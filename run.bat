@echo off
REM DocMind launcher - uses Python 3.12 (faiss/torch compatible)
REM First time: install deps + create .env from template

if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo Created .env - open it and paste your GROQ_API_KEY.
    )
)

py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
