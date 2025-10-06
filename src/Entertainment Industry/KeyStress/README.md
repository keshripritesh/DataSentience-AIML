# 🔐 KeyStress — Keystroke-Based Stress & Fatigue Detector

**Domain:** Behavioral AI / Digital Biometrics  
**What it does:** Monitors typing behavior (speed, errors, pauses) and trains a lightweight ML model to estimate **stress or fatigue levels**.  
**Data:** Collected directly from **user typing logs** (no external dataset required).

## 🎯 Motivation

Typing patterns subtly change under stress or fatigue:

* Higher **error rates**
* More frequent **pauses / hesitations**
* Faster or erratic **keystroke dynamics**

This project aims to capture those changes and map them to stress levels with **on-device, privacy-friendly ML models**.

## 🗂️ Project Structure

```
keystress/
│
├─ capture/
│  ├─ keylogger.py        # Non-intrusive keystroke logger (OS hooks)
│  └─ session_recorder.py # Records typing sessions with timestamps
│
├─ features/
│  ├─ extractor.py        # Compute speed, pauses, error rate, variability
│  ├─ stress_indicators.py# Aggregate features into stress-relevant signals
│
├─ ml/
│  ├─ dataset.py          # Collects & formats logs into training samples
│  ├─ model.py            # ML classifier/regressor (LogReg, RandomForest, LSTM)
│  ├─ train.py            # Training loop
│  └─ evaluate.py         # Metrics & cross-validation
│
├─ ui/
│  ├─ cli.py              # Start/stop monitoring, real-time stress indicator
│  └─ dashboard.py        # Streamlit dashboard for visualization
│
├─ data/
│  └─ logs/               # User keystroke sessions (local only)
│
├─ tests/
│  └─ test_features.py
│
├─ README.md
└─ requirements.txt
```

## 🧠 Features Captured

1. **Typing Speed**
   * Keys per minute, average latency between keystrokes

2. **Error Metrics**
   * Backspace/delete frequency
   * Corrections per 100 characters

3. **Pauses & Burstiness**
   * Distribution of typing intervals
   * Longer pauses may indicate distraction or fatigue

4. **Variability**
   * Standard deviation of inter-key intervals
   * Increased variance → possible stress

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Start a typing session:**
   ```bash
   python ui/cli.py --session study_notes
   ```

2. **Train a model:**
   ```bash
   python ml/train.py --data data/logs --model randomforest
   ```

3. **Launch dashboard:**
   ```bash
   python ui/dashboard.py
   ```

## 🧪 Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

## 📊 Example Workflow

1. **Data Collection:**
   ```bash
   python capture/keylogger.py --session study_notes
   # User types naturally, data saved as JSON log
   ```

2. **Feature Extraction:**
   ```
   Session: study_notes
   - Avg speed: 180 CPM
   - Backspace rate: 12%
   - Avg pause: 450ms
   - Variability: High
   ```

3. **Model Training:**
   ```bash
   python ml/train.py --data data/logs --model randomforest
   ```

4. **Real-Time Feedback (Dashboard):**
   * Gauge: "Stress Level: 🔴 High"
   * Trend chart of stress vs. session duration

## 📈 Metrics

* Classification: Accuracy, F1-score, ROC-AUC
* Regression (if using scale): RMSE, R²
* Longitudinal: Does the model capture **fatigue buildup** across sessions?

## 🛠️ Tech Stack

* **Data Capture:** `pynput`, `keyboard` (Python keystroke hooks)
* **Feature Processing:** `numpy`, `pandas`
* **ML Models:** `scikit-learn`, `torch` (for LSTM)
* **UI:** `streamlit`, `matplotlib`

## 🔮 Future Enhancements

* **Real-time monitoring:** Notify user when stress exceeds threshold
* **Personalized models:** Adapt baseline per user
* **Cross-device support:** Desktop + mobile keystroke logging
* **Integration:** Combine with mouse dynamics / gaze tracking for richer signals

## 📝 License

MIT License - see LICENSE file for details
