https://student-performance-predictions-nj8h.onrender.com
# 🎓 Student Performance Classifier (Flask + Render)

A full-stack, responsive web interface built with **Flask**, **JavaScript**, and **Scikit-learn** to predict student performance classifications based on demographic and academic metrics. Styled with dynamic CSS theme switching and interactive CSS animations.

---

## 🚀 Features

- **Integrated Frontend & Backend:** Serves HTML/CSS/JS dynamically within a single `app.py` file.
- **Dynamic Themes:** Toggle between **Light**, **Dark**, and **Cyberpunk** modes on the fly.
- **Asynchronous Processing:** Built using native `fetch()` calls for smooth, non-reloading submission animations.
- **Render Ready:** Configured with `gunicorn` and environment port bindings for quick cloud deployment.

---

## 🛠️ Machine Learning Model Features

The underlying `svc_model.pkl` accepts **9 input variables** in the following order:

| Feature Name | Description | Type / Input Range |
| :--- | :--- | :--- |
| `gender` | Student gender | `0` (Female), `1` (Male) |
| `age` | Age in years | Numeric (e.g., `16`) |
| `study_hours` | Weekly self-study hours | Numeric (e.g., `15.0`) |
| `attendance` | Class attendance percentage | Range `0.0` – `100.0` |
| `parent_edu` | Parent education level | `0` (Uneducated/Primary) to `3` (Post-Graduate) |
| `internet` | Internet access at home | `1` (Yes), `0` (No) |
| `extracurricular` | Participation in activities | `1` (Yes), `0` (No) |
| `prev_score` | Previous academic score | Range `0.0` – `100.0` |
| `final_score` | Current test score / average | Range `0.0` – `100.0` |

---

## 📁 Repository Structure

```text
├── app.py              # Main Flask application and inline HTML template
├── svc_model.pkl       # Trained Support Vector Classifier model
├── requirements.txt    # Required Python packages
└── README.md           # Project documentation
💻 Local Setup & Execution
Clone the Repository:

Bash
git clone [https://github.com/your-username/student-performance-classifier.git](https://github.com/your-username/student-performance-classifier.git)
cd student-performance-classifier
Create and Activate a Virtual Environment:

Bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install Dependencies:

Bash
pip install -r requirements.txt
Run the App Locally:

Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000.

🌐 Deploying to Render
Push your code to a GitHub repository (make sure app.py, svc_model.pkl, and requirements.txt are at the root level).

Log in to Render.

Click New + > Web Service.

Connect your GitHub repository.

Fill in the deployment details:

Environment: Python 3

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

Click Create Web Service.
