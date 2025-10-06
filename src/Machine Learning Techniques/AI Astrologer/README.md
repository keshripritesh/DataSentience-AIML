# 🔮 AI Astrologer

AI-powered astrology prediction app built with Streamlit and Mistral AI. Enter your birth details, ask personalized questions, and receive insightful, AI-generated predictions.

---

## 🚀 Features
- Input: Name, Date of Birth, Time of Birth (AM/PM), and Place of Birth.
- Ask personalized questions for tailored predictions.
- Clean, responsive UI with Streamlit.
- Powered by Mistral AI API for high-quality responses.

---

## 🧰 Tech Stack
- Streamlit (UI)
- Mistral AI API (LLM)
- Python 3.9+ (recommended)
- python-dotenv for environment variables

---

## 📺 Demo
[🔗 Demo Video Link will be added here]

---

## 🛠️ Getting Started

### 1) Clone the repository
```bash
git clone https://github.com/yourusername/ai-astrologer.git
cd ai-astrologer
```

### 2) Create and activate a virtual environment

- Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\activate
```

- macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Set up environment variables

Create a `.env` file in the project root (or copy `.env.example` to `.env`) and add your Mistral API key:
```ini
MISTRAL_API_KEY=your_api_key_here
```

Optional (alternative): Set the key in your shell session
- macOS/Linux:
```bash
export MISTRAL_API_KEY=your_api_key_here
```
- Windows (PowerShell):
```powershell
$Env:MISTRAL_API_KEY="your_api_key_here"
```

### 5) Run the app
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal (typically http://localhost:8501).

---

## 🧑‍💻 Usage
1. Open the app in your browser.
2. Enter:
   - Name
   - Date of Birth
   - Time of Birth (with AM/PM)
   - Place of Birth
3. Ask your question (e.g., career, relationships, finance, health).
4. Submit to receive your AI-generated prediction.

Tips:
- Include the city and country for Place of Birth for better context.
- If you know your exact birth time, include it for improved accuracy.

---

## 📂 Project Structure
```
ai-astrologer/
├─ app.py              # Main Streamlit app
├─ requirements.txt    # Dependencies
├─ .env.example        # Example environment file
└─ README.md           # Documentation
```

---

## 🧪 Troubleshooting
- Mistral API errors (401/403):
  - Verify your `MISTRAL_API_KEY` is correct and active.
  - Ensure `.env` is in the project root and loaded before app code runs.
- Module not found (e.g., streamlit):
  - Reinstall dependencies: `pip install -r requirements.txt`
  - Ensure your virtual environment is activated.
- Port already in use:
  - Run on another port: `streamlit run app.py --server.port 8502`
- .env not loading:
  - Confirm `python-dotenv` is installed and your app loads environment variables early.

Security note: Never commit your `.env` file or API keys to version control.

---

## ⚠️ Disclaimer
This app provides AI-generated astrological insights for entertainment and educational purposes only. It should not be used as a substitute for professional advice.

---

## 🙌 Acknowledgements
- [Streamlit](https://streamlit.io/)
- [Mistral AI](https://mistral.ai/)

---

## ✨ Developed By
Pritesh Keshri