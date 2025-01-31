#DEA evaluates the efficiency of Decision-Making Units (DMUs) (e.g., employees) 
# by comparing their ability to convert inputs into outputs relative to the "best practice" frontier (most efficient peers). 
# The simplest DEA model is the CCR model (Charnes-Cooper-Rhodes), which assumes constant returns to scale (CRS).
import json
import pandas as pd
import pulp as lp
import os

# Use the absolute path for the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), '../../database/employee_data.json')
with open(json_file_path, 'r') as f:
    data = json.load(f)
# Preprocess data: calculate weighted outputs and input cost
employees = []
for emp in data['employees']:
    # Calculate weighted outputs
    email_out = emp['emails_sent'] * 0.1
    code_out = emp['code_commits'] * 0.6
    doc_out = emp['documents_submitted'] * 0.6
    meeting_out = emp['customer_meeting_hours'] * 0.5
    
    # Calculate input (unit_cost * hours_worked)
    input_cost = emp['unit_cost'] * emp['hours_worked']
    
    employees.append({
        'name': emp['name'],
        'input': input_cost,
        'outputs': [email_out, code_out, doc_out, meeting_out]
    })
# Convert to DataFrame
df = pd.DataFrame(employees).set_index('name')
# Extract DMUs (employee names)
dmus = df.index.tolist()
num_outputs = 4  # email, code, doc, meeting
# Store efficiency scores
efficiency_scores = {}
# Solve output-oriented VRS DEA for each DMU
for target_dmu in dmus:
    # Get target DMU's input and outputs
    target_input = df.loc[target_dmu, 'input']
    target_outputs = df.loc[target_dmu, 'outputs']
    
    # Create LP problem
    prob = lp.LpProblem(f'DEA_Output_VRS_{target_dmu}', lp.LpMaximize)
    
    # Variables: phi (output expansion factor) and lambdas (intensity variables)
    phi = lp.LpVariable('phi', lowBound=1)  # phi >=1 in output orientation
    lambdas = lp.LpVariable.dicts('lambda', dmus, lowBound=0)
    
    # Objective: Maximize phi
    prob += phi
    
    # Constraints
    # 1. Input constraint: Sum(lambda[j] * input[j]) <= target_input
    prob += lp.lpSum(lambdas[j] * df.loc[j, 'input'] for j in dmus) <= target_input
    
    # 2. Output constraints: Sum(lambda[j] * output[j][k]) >= phi * target_output[k]
    for k in range(num_outputs):
        prob += lp.lpSum(lambdas[j] * df.loc[j, 'outputs'][k] for j in dmus) >= phi * target_outputs[k]
    
    # 3. VRS constraint: Sum(lambdas) = 1
    prob += lp.lpSum(lambdas[j] for j in dmus) == 1
    
    # Solve
    prob.solve(lp.PULP_CBC_CMD(msg=False))
    
    # Efficiency score = 1/phi (to keep score <=1)
    if phi.varValue is not None and phi.varValue > 0:
        efficiency = 1 / phi.varValue

    else:
        efficiency = 0  # Infeasible or error
    efficiency_scores[target_dmu] = efficiency
# Print results
# print("Efficiency Scores (Output-Oriented VRS DEA):")
# for dmu, score in efficiency_scores.items():
#     print(f"{dmu}: {score:.3f}")
    
# Create a list with only the scores
score_list = list(efficiency_scores.values())
# Print the list of scores
# print("List of scores:", score_list)
def get_efficiency_scores():
    return score_list