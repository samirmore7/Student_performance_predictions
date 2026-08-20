import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ==========================================
# 1. ROBUST MODEL INITIALIZATION (SVC_model.pkl)
# ==========================================
model = None
base_dir = os.path.dirname(os.path.abspath(__file__))

candidate_paths = [
    os.path.join(base_dir, 'SVC_model.pkl'),
    os.path.join(os.getcwd(), 'SVC_model.pkl'),
    'SVC_model.pkl'
]

for path in candidate_paths:
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            break
        except Exception:
            continue

# ==========================================
# 2. EMBEDDED UI & ADVANCED ANALYTICS DASHBOARD
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Matrix & Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
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
            --bar-bg: #e2e8f0;
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
            --bar-bg: #374151;
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
            --bar-bg: #2d124d;
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
            max-width: 900px;
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

        /* ANALYTICS CONTAINER */
        #analytics-box {
            display: none;
            margin-top: 2.5rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
            animation: fadeIn 0.5s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .kpi-card {
            background: var(--surface-subtle);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem;
            text-align: center;
        }

        .kpi-title {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .kpi-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--primary);
            margin-top: 0.25rem;
        }

        .metric-row {
            margin-bottom: 1rem;
        }

        .metric-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .progress-bar-bg {
            width: 100%;
            height: 10px;
            background-color: var(--bar-bg);
            border-radius: 5px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background-color: var(--primary);
            width: 0%;
            transition: width 0.8s ease-in-out;
        }

        .insights-list {
            list-style: none;
            margin-top: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .insight-item {
            background: var(--surface-subtle);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-left: 4px solid var(--primary);
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
            <p class="subtitle">SVC inference engine & full diagnostic analytics suite</p>
        </div>
        <select class="theme-select" id="themeSelector" onchange="applyTheme(this.value)">
            <option value="dark">🌙 Dark Slate</option>
            <option value="light">☀️ Clean Light</option>
            <option value="cyberpunk">⚡ Cyberpunk Neon</option>
        </select>
    </header>

    {% if not model_loaded %}
        <div style="color: #ef4444; background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #ef4444;">
            ⚠️ <strong>Missing Model File:</strong> Ensure <code>SVC_model.pkl</code> exists in the root directory.
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
            <span id="btnText">🔮 Run Classification & Analytics</span>
        </button>
    </form>

    <!-- EXPANDED ANALYTICS SECTION -->
    <div id="analytics-box">
        <h2 style="font-size: 1.25rem; font-weight: 800; margin-bottom: 1rem;">📊 Performance & Diagnostic Analytics</h2>
        
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Predicted Class</div>
                <div class="kpi-value" id="kpi-class">-</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Decision Confidence</div>
                <div class="kpi-value" id="kpi-conf">-</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Risk Index</div>
                <div class="kpi-value" id="kpi-risk">-</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Academic Momentum</div>
                <div class="kpi-value" id="kpi-delta">-</div>
            </div>
        </div>

        <div style="margin-top: 1.5rem;">
            <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 1rem; color: var(--text-secondary);">BENCHMARK VS TARGETS</h3>
            
            <div class="metric-row">
                <div class="metric-header">
                    <span>Attendance Benchmark</span>
                    <span id="bar-att-val">0%</span>
                </div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="bar-att"></div></div>
            </div>

            <div class="metric-row">
                <div class="metric-header">
                    <span>Weekly Study Target (Max 40 hrs base)</span>
                    <span id="bar-study-val">0 hrs</span>
                </div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="bar-study"></div></div>
            </div>

            <div class="metric-row">
                <div class="metric-header">
                    <span>Overall Academic Score</span>
                    <span id="bar-score-val">0%</span>
                </div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="bar-score"></div></div>
            </div>
        </div>

        <div style="margin-top: 1.5rem;">
            <h3 style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-secondary);">INFLUENCE DRIVERS & DIAGNOSTICS</h3>
            <ul class="insights-list" id="insights-container"></ul>
        </div>
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
        const analyticsBox = document.getElementById('analytics-box');

        spinner.style.display = 'inline-block';
        btnText.innerText = 'Analyzing Metrics...';
        btn.disabled = true;

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
                // Populate KPIs
                document.getElementById('kpi-class').innerText = 'Class ' + data.prediction;
                document.getElementById('kpi-conf').innerText = data.analytics.confidence + '%';
                document.getElementById('kpi-risk').innerText = data.analytics.risk_index;
                document.getElementById('kpi-delta').innerText = (data.analytics.score_delta > 0 ? '+' : '') + data.analytics.score_delta + ' pts';

                // Populate Progress Bars
                document.getElementById('bar-att').style.width = Math.min(data.analytics.attendance, 100) + '%';
                document.getElementById('bar-att-val').innerText = data.analytics.attendance + '%';

                document.getElementById('bar-study').style.width = Math.min((data.analytics.study_hours / 40) * 100, 100) + '%';
                document.getElementById('bar-study-val').innerText = data.analytics.study_hours + ' hrs/wk';

                document.getElementById('bar-score').style.width = Math.min(data.analytics.final_score, 100) + '%';
                document.getElementById('bar-score-val').innerText = data.analytics.final_score + '%';

                // Populate Insights List
                const list = document.getElementById('insights-container');
                list.innerHTML = '';
                data.analytics.insights.forEach(item => {
                    const li = document.createElement('li');
                    li.className = 'insight-item';
                    li.innerHTML = item;
                    list.appendChild(li);
                });

                analyticsBox.style.display = 'block';
            } else {
                alert('Prediction Error: ' + data.error);
            }
        } catch (err) {
            alert('Failed to connect to the backend engine.');
        } finally {
            spinner.style.display = 'none';
            btnText.innerText = '🔮 Run Classification & Analytics';
            btn.disabled = false;
        }
    });
</script>
</body>
</html>
"""

# ==========================================
# 3. ROUTE ENDPOINTS & ANALYTICS CALCULATION
# ==========================================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, model_loaded=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': 'Model file (SVC_model.pkl) is unavailable.'})
    
    try:
        data = request.json
        gender = int(data['gender'])
        age = float(data['age'])
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        parent_edu = int(data['parent_edu'])
        internet = int(data['internet'])
        extracurricular = int(data['extracurricular'])
        prev_score = float(data['prev_score'])
        final_score = float(data['final_score'])

        features = np.array([[
            gender, age, study_hours, attendance, parent_edu,
            internet, extracurricular, prev_score, final_score
        ]])
        
        prediction = model.predict(features)[0]

        # -------------------------------------------------------------
        # CALCULATE COMPREHENSIVE ANALYTICS
        # -------------------------------------------------------------
        # 1. Decision Boundary Distance -> Estimated Confidence
        decision_val = float(model.decision_function(features)[0])
        # Logistic sigmoid mapping of distance to proxy confidence (50% - 99%)
        confidence = round((1 / (1 + np.exp(-abs(decision_val)))) * 100, 1)

        # 2. Score Momentum (Change in performance)
        score_delta = round(final_score - prev_score, 1)

        # 3. Risk Level Assessment
        if attendance < 65 or final_score < 50:
            risk_index = "High"
        elif attendance < 80 or final_score < 68:
            risk_index = "Moderate"
        else:
            risk_index = "Low"

        # 4. Actionable Diagnostic Insights
        insights = []
        if attendance < 75:
            insights.append("⚠️ <strong>Attendance Warning:</strong> Attendance rate is sub-optimal; consistent class attendance strongly stabilizes SVC classifications.")
        else:
            insights.append("✅ <strong>Strong Attendance:</strong> High attendance acts as a major positive anchor for classification.")

        if study_hours >= 15:
            insights.append(f"✅ <strong>Dedicated Study Time:</strong> {study_hours} hrs/week places the student in the upper performance tier.")
        else:
            insights.append(f"📌 <strong>Study Opportunity:</strong> Increasing study duration beyond {study_hours} hrs/week could improve academic trajectory.")

        if score_delta > 0:
            insights.append(f"📈 <strong>Positive Growth:</strong> Gained {score_delta} points compared to previous test performance.")
        elif score_delta < 0:
            insights.append(f"📉 <strong>Performance Dip:</strong> Dropped {abs(score_delta)} points relative to the previous benchmark.")
        else:
            insights.append("⚖️ <strong>Stable Performance:</strong> Final score directly mirrors prior academic baseline.")

        if internet == 1 and extracurricular == 1:
            insights.append("🌟 <strong>Holistic Profile:</strong> High resource access and extracurricular engagement complement direct metrics.")

        analytics_payload = {
            'confidence': confidence,
            'risk_index': risk_index,
            'score_delta': score_delta,
            'attendance': attendance,
            'study_hours': study_hours,
            'final_score': final_score,
            'insights': insights
        }

        return jsonify({
            'success': True,
            'prediction': str(prediction),
            'analytics': analytics_payload
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
