import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ==========================================
# 1. MODEL INITIALIZATION (UNIVERSAL PATH)
# ==========================================
# Resolves model path whether deployed on Render (root) or Vercel (/api subfolder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
    os.path.join(CURRENT_DIR, 'svc_model.pkl'),
    os.path.join(os.path.dirname(CURRENT_DIR), 'svc_model.pkl'),
    'svc_model.pkl'
]

model = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            break
        except Exception:
            continue

# ==========================================
# 2. EMBEDDED UI & ANIMATED THEMES
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Matrix</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
        /* THEME DEFINITIONS */
        :root[data-theme="light"] {
            --bg-canvas: #f8fafc;
            --surface: #ffffff;
            --surface-subtle: #f1f5f9;
            --border-color: #e2e8f0;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --accent-glow: rgba(79, 70, 229, 0.15);
            --card-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
        }

        :root[data-theme="dark"] {
            --bg-canvas: #090d16;
            --surface: #111827;
            --surface-subtle: #1f2937;
            --border-color: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent-glow: rgba(99, 102, 241, 0.25);
            --card-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        :root[data-theme="cyberpunk"] {
            --bg-canvas: #05050d;
            --surface: #0f0a1c;
            --surface-subtle: #1b1130;
            --border-color: #ff007f;
            --text-primary: #00f0ff;
            --text-secondary: #d946ef;
            --primary: #ff007f;
            --primary-hover: #d6006b;
            --accent-glow: rgba(255, 0, 127, 0.4);
            --card-shadow: 0 0 25px rgba(255, 0, 127, 0.25), 0 0 15px rgba(0, 240, 255, 0.2);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
        }

        body {
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2.5rem 1rem;
        }

        .dashboard-card {
            width: 100%;
            max-width: 820px;
            background-color: var(--surface);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            box-shadow: var(--card-shadow);
            padding: 2.5rem;
            animation: cardEntrance 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes cardEntrance {
            from { opacity: 0; transform: translateY(20px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }

        h1 {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        p.subtitle {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }

        .theme-select {
            padding: 0.6rem 1rem;
            background: var(--surface-subtle);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            outline: none;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .field-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-secondary);
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            background-color: var(--surface-subtle);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .btn-predict {
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 12px;
            background-color: var(--primary);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
        }

        .btn-predict:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }

        .btn-predict:active {
            transform: translateY(1px);
        }

        /* RESULT DISPLAY & ANIMATION */
        #result-box {
            display: none;
            margin-top: 2rem;
            padding: 1.75rem;
            background-color: var(--surface-subtle);
            border: 1px dashed var(--primary);
            border-radius: 14px;
            text-align: center;
            transform: scale(0.95);
            opacity: 0;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }

        #result-box.visible {
            display: block;
            transform: scale(1);
            opacity: 1;
        }

        #result-value {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--primary);
            margin-top: 0.25rem;
        }

        .spinner {
            width: 18px;
            height: 18px;
            border: 2px solid #ffffff;
            border-bottom-color: transparent;
            border-radius: 50%;
            display: none;
            animation: rotation 1s linear infinite;
        }

        @keyframes rotation {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="dashboard-card">
    <header>
        <div>
            <h1>🎓 Student Performance Matrix</h1>
            <p class="subtitle">SVC inference engine for academic trajectory classification</p>
        </div>
        <select class="theme-select" id="themeSelector" onchange="applyTheme(this.value)">
            <option value="dark">🌙 Dark Slate</option>
            <option value="light">☀️ Clean Light</option>
            <option value="cyberpunk">⚡ Cyberpunk Neon</option>
        </select>
    </header>

    {% if not model_loaded %}
        <div style="color: #ef4444; background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #ef4444;">
            ⚠️ <strong>Missing Model File:</strong> Ensure <code>svc_model.pkl</code> exists in the root directory.
        </div>
    {% endif %}

    <form id="evaluationForm">
        <div class="form-grid">
            <div class="field-group">
                <label>Gender</label>
                <select name="gender">
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>
            <div class="field-group">
                <label>Age</label>
                <input type="number" name="age" min="10" max="100" value="16" required>
            </div>
            <div class="field-group">
                <label>Study Hours / Week</label>
                <input type="number" name="study_hours" step="0.1" value="15.0" required>
            </div>
            <div class="field-group">
                <label>Attendance Rate (%)</label>
                <input type="number" name="attendance" min="0" max="100" step="0.1" value="85.0" required>
            </div>
            <div class="field-group">
                <label>Parent Education</label>
                <select name="parent_edu">
                    <option value="0">Primary / High School</option>
                    <option value="1">Some College</option>
                    <option value="2">Graduate</option>
                    <option value="3">Post-Graduate</option>
                </select>
            </div>
            <div class="field-group">
                <label>Internet Access</label>
                <select name="internet">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="field-group">
                <label>Extracurriculars</label>
                <select name="extracurricular">
                    <option value="1">Active</option>
                    <option value="0">None</option>
                </select>
            </div>
            <div class="field-group">
                <label>Previous Score</label>
                <input type="number" name="prev_score" min="0" max="100" step="0.1" value="70.0" required>
            </div>
            <div class="field-group">
                <label>Current / Final Score</label>
                <input type="number" name="final_score" min="0" max="100" step="0.1" value="75.0" required>
            </div>
        </div>

        <button type="submit" class="btn-predict" id="submitBtn">
            <span class="spinner" id="btnSpinner"></span>
            <span id="btnText">🔮 Run Classification Prediction</span>
        </button>
    </form>

    <div id="result-box">
        <p style="color: var(--text-secondary); font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">Predicted Performance Classification</p>
        <div id="result-value">-</div>
    </div>
</div>

<script>
    function applyTheme(name) {
        document.documentElement.setAttribute('data-theme', name);
    }

    document.getElementById('evaluationForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const btn = document.getElementById('submitBtn');
        const spinner = document.getElementById('btnSpinner');
        const btnText = document.getElementById('btnText');
        const resultBox = document.getElementById('result-box');
        const resultVal = document.getElementById('result-value');

        // Loading animation state
        spinner.style.display = 'inline-block';
        btnText.innerText = 'Calculating Matrix...';
        btn.disabled = true;
        resultBox.classList.remove('visible');

        const formData = new FormData(e.target);
        const payload = Object.fromEntries(formData.entries());

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success) {
                resultVal.innerText = 'Class: ' + data.prediction;
                resultBox.style.display = 'block';
                setTimeout(() => resultBox.classList.add('visible'), 30);
            } else {
                alert('Prediction Error: ' + data.error);
            }
        } catch (err) {
            alert('Failed to connect to the prediction server.');
        } finally {
            spinner.style.display = 'none';
            btnText.innerText = '🔮 Run Classification Prediction';
            btn.disabled = false;
        }
    });
</script>
</body>
</html>
"""

# ==========================================
# 3. ROUTE ENDPOINTS
# ==========================================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, model_loaded=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': 'Model pickle file is unavailable.'})
    
    try:
        data = request.json
        # Feature array configured to match trained SVC layout
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

# Dynamic server invocation for container / cloud hosting
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
