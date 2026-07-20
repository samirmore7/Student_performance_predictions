import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- Load the SVC Model Safely ---
MODEL_PATH = 'svc_model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

# --- Single HTML Template with Built-in CSS & Animations ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        /* CSS variables for Theme Management */
        :root[data-theme="light"] {
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #4b5563;
            --accent: #4f46e5;
            --accent-hover: #4338ca;
            --border: #e5e7eb;
            --card-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        }

        :root[data-theme="dark"] {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --border: #334155;
            --card-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        }

        :root[data-theme="cyberpunk"] {
            --bg-color: #0d0221;
            --card-bg: #140132;
            --text-main: #00ffff;
            --text-muted: #ff007f;
            --accent: #ff007f;
            --accent-hover: #bc005b;
            --border: #00ffff;
            --card-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            transition: background-color 0.4s ease, color 0.4s ease, border-color 0.4s ease;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: var(--card-shadow);
            border: 1px solid var(--border);
            padding: 2.5rem;
            position: relative;
            overflow: hidden;
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
        }

        h1 { font-size: 1.75rem; font-weight: 700; }
        p.subtitle { color: var(--text-muted); font-size: 0.95rem; margin-top: 0.25rem; }

        .theme-selector select {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: var(--card-bg);
            color: var(--text-main);
            border: 1px solid var(--border);
            font-weight: 500;
            cursor: pointer;
            outline: none;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-main);
        }

        input, select {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: var(--bg-color);
            color: var(--text-main);
            border: 1px solid var(--border);
            outline: none;
            font-size: 0.95rem;
        }

        input:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .btn-submit {
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 8px;
            background: var(--accent);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.1s ease, background-color 0.2s ease;
        }

        .btn-submit:hover { background-color: var(--accent-hover); }
        .btn-submit:active { transform: scale(0.98); }

        /* Animation overlay for calculation results */
        #result-container {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 12px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px dashed var(--accent);
            text-align: center;
            display: none;
            opacity: 0;
            transform: scale(0.95);
            transition: all 0.4s ease;
        }

        #result-container.show {
            display: block;
            opacity: 1;
            transform: scale(1);
        }

        #result-value {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--accent);
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <div>
            <h1>🎓 Student Performance Matrix</h1>
            <p class="subtitle">Classify predictive metrics using your SVC Machine Learning pipeline</p>
        </div>
        <div class="theme-selector">
            <select id="themeSelect" onchange="changeTheme(this.value)">
                <option value="light">☀️ Light</option>
                <option value="dark">🌙 Dark</option>
                <option value="cyberpunk">🌌 Cyberpunk</option>
            </select>
        </div>
    </header>

    {% if error_msg %}
        <div style="color: red; padding: 1rem; border: 1px solid red; border-radius: 8px; margin-bottom: 1rem;">
            {{ error_msg }}
        </div>
    {% endif %}

    <form id="predictionForm">
        <div class="form-grid">
            <div class="form-group">
                <label>Gender</label>
                <select name="gender">
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>
            <div class="form-group">
                <label>Age</label>
                <input type="number" name="age" min="10" max="100" value="16" required>
            </div>
            <div class="form-group">
                <label>Study Hours per Week</label>
                <input type="number" name="study_hours" step="0.1" value="15.0" required>
            </div>
            <div class="form-group">
                <label>Attendance Rate (%)</label>
                <input type="number" name="attendance" step="0.1" min="0" max="100" value="85.0" required>
            </div>
            <div class="form-group">
                <label>Parent Education Level</label>
                <select name="parent_edu">
                    <option value="0">Uneducated/Primary</option>
                    <option value="1">High School</option>
                    <option value="2">Graduate</option>
                    <option value="3">Post-Graduate</option>
                </select>
            </div>
            <div class="form-group">
                <label>Has Internet Access?</label>
                <select name="internet">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label>Participates in Extracurriculars?</label>
                <select name="extracurricular">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label>Previous Score</label>
                <input type="number" name="prev_score" step="0.1" min="0" max="100" value="70.0" required>
            </div>
            <div class="form-group">
                <label>Final Test Score / Current Average</label>
                <input type="number" name="final_score" step="0.1" min="0" max="100" value="75.0" required>
            </div>
        </div>

        <button type="submit" class="btn-submit">🔮 Run Analysis Prediction</button>
    </form>

    <div id="result-container">
        <p>Target Output Class Assessment</p>
        <div id="result-value">-</div>
    </div>
</div>

<script>
    function changeTheme(themeName) {
        document.documentElement.setAttribute('data-theme', themeName);
    }

    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());
        
        const resContainer = document.getElementById('result-container');
        const resValue = document.getElementById('result-value');
        
        resContainer.style.display = 'none';
        resContainer.classList.remove('show');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            if(result.success) {
                resValue.innerText = "Class " + result.prediction;
                resContainer.style.display = 'block';
                setTimeout(() => resContainer.classList.add('show'), 50);
            } else {
                alert("Error: " + result.error);
            }
        } catch (err) {
            alert("Failed to connect to backend engine processing endpoint.");
        }
    });
</script>
</body>
</html>
"""

@app.route('/')
def home():
    error_msg = None if model else "⚠️ Missing 'svc_model.pkl' inside your base root deployment path."
    return render_template_string(HTML_TEMPLATE, error_msg=error_msg)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': 'Model pickle file uninitialized.'})
    
    try:
        data = request.json
        # Extract features ensuring sequence order strictly reflects pickle layout
        features = np.array([[
            int(data['gender']),
            float(data['age']),
            float(data['study_hours']),
            float(data['attendance']),
            int(data['parent_edu']),
            int(data['internet']),
            int(data['extracurricular']),
            float(data['prev_score']),
            float(data['final_score'])
        ]])
        
        prediction = model.predict(features)[0]
        return jsonify({'success': True, 'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Dynamic binding for web engines like Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
