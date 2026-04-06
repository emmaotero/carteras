# Deploy — Portfolio Tracker (Streamlit)

## Archivos del proyecto
```
portfolio-tracker/
├── app.py                    ← app principal
├── requirements.txt          ← dependencias
├── .gitignore
└── .streamlit/
    └── secrets.toml          ← credenciales (NO subir a GitHub)
```

---

## Paso 1: Supabase (si ya lo tenés, saltá al paso 2)

1. Entrá a https://supabase.com → creá proyecto
2. SQL Editor → ejecutá el script `supabase-schema.sql` anterior
3. Project Settings → API → copiá **URL** y **anon key**

---

## Paso 2: GitHub

1. Entrá a https://github.com → **New repository**
2. Nombre: `portfolio-tracker`  |  Visibilidad: **Public**
3. **Create repository**
4. Clic en **uploading an existing file**
5. Subí estos archivos: `app.py`, `requirements.txt`, `.gitignore`
   ⚠ NO subas `.streamlit/secrets.toml` (tiene tus credenciales)
6. **Commit changes**

---

## Paso 3: Streamlit Cloud

1. Entrá a https://share.streamlit.io
2. Creá cuenta con GitHub (botón "Continue with GitHub")
3. Clic en **New app**
4. Repository: `tu-usuario/portfolio-tracker`
5. Branch: `main`
6. Main file path: `app.py`
7. Clic en **Advanced settings** → **Secrets**
8. Pegá esto (con tus valores reales):
   ```
   SUPABASE_URL = "https://tu-proyecto.supabase.co"
   SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIs..."
   ```
9. **Save** → **Deploy**
10. En ~2 minutos tenés tu URL pública

---

## Actualizar la app

Editá `app.py` directamente en GitHub → commit → Streamlit redeploya solo.

---

## Probar localmente en Colab (opcional)

```python
# En una celda de Colab:
!pip install streamlit supabase yfinance pandas-ta plotly -q
!pip install pyngrok -q

import subprocess, threading
from pyngrok import ngrok

# Crear secrets
import os
os.makedirs('.streamlit', exist_ok=True)
with open('.streamlit/secrets.toml','w') as f:
    f.write('SUPABASE_URL = "https://tu-proyecto.supabase.co"\n')
    f.write('SUPABASE_KEY = "tu-key"\n')

# Lanzar
public_url = ngrok.connect(8501)
print("URL:", public_url)
threading.Thread(target=lambda: subprocess.run(["streamlit","run","app.py","--server.port=8501"])).start()
```

---

## Tickers argentinos soportados

| Ticker local | Yahoo Finance |
|---|---|
| YPF | YPF |
| GGAL | GGAL |
| BMA | BMA |
| PAMP | PAM |
| TECO2 | TEO |
| CEPU | CEPU |
| LOMA | LOMA.BA |
| TXAR | TXAR.BA |
| BBAR | BBAR.BA |

Para bonos (AL30, GD30) que no están en Yahoo → cargá precio manual en la ficha del cliente.
