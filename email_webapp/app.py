import os, sys
from flask import Flask, render_template, request, jsonify
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.fine_tuning.SFT import SupervisedFineTraining

app = Flask(__name__)
sft = SupervisedFineTraining()
sft.tuned_model_name = os.path.join(root_path, "bert-base-uncased_fine_tuned")
sft.load_tuned_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    email_text = data.get('email_text', '')

    output_scores = sft.infer(email_text)
    
    return jsonify({
        'english_proficiency': float(output_scores["english_proficiency"]),
        'clarity': float(output_scores["clarity"]),
        'technical_depth': float(output_scores["technical_depth"]),
        'aggressivity': float(output_scores["aggressivity"]),
        'empathy': float(output_scores["empathy"]),
    })

if __name__ == '__main__':
    app.run(debug=True)