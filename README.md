# ProjekBunyi

requirements.txt
``` bash
streamlit>=1.28.0
numpy>=1.24.3
scikit-learn>=1.3.0
joblib>=1.3.2
matplotlib>=3.7.2
pydub>=0.25.1
soundfile>=0.12.1
audioread>=3.0.0
```

Buat Virtual Environment di VS Code
# 1. Hapus virtual environment lama jika ada
``` bash
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
```

# 2. Buat virtual environment baru
``` bash
python -m venv venv
```

# 3. Aktifkan virtual environment
# Untuk PowerShell di VS Code, biasanya otomatis aktif
# Jika tidak, coba:

``` bash
.\venv\Scripts\Activate.ps1
```

# 4. Tampilkan path Python untuk memastikan

``` bash
python --version
where python
```

Install Dependencies Satu per Satu

# 1. Upgrade pip
``` bash
python -m pip install --upgrade pip
```
# 2. Install setuptools dan wheel dulu
``` bash
pip install setuptools wheel
```
# 3. Install packages satu per satu
``` bash
pip install numpy
pip install scikit-learn
pip install joblib
pip install matplotlib
pip install streamlit
```
# 4. Untuk audio processing, install yang ringan
``` bash
pip install pydub
pip install soundfile
pip install streamlit librosa joblib matplotlib numpy soundfile
```

Jalankan di VS Code
# Pastikan virtual environment aktif
``` bash
.\venv\Scripts\Activate.ps1
```
# Jalankan Streamlit
``` bash
streamlit run app.py
```

