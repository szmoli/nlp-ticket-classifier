from flask import Flask, render_template, request, redirect, url_for
import db
import os
from classify import team, load_models, info, train

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

try:
    ft_model, clf, metrics = load_models()
    print("[INFO] ML models loaded.")
except Exception as e:
    print(f"[WARN] ML models not loaded: {e}")
    ft_model, clf, metrics = train()

db.initialize()

@app.route('/')
def index():
    tickets = db.get_all_tickets()
    return render_template('index.html', tickets=tickets)

@app.route('/info', methods=['GET'])
def view_info():
    if ft_model is not None and clf is not None and metrics is not None:
        inf = info(ft_model, clf, metrics)
    else:
        inf = {}
    
    print(f"[INFO] Model info: {inf}")
    return render_template('info.html', info=inf)

@app.route('/new', methods=['GET', 'POST'])
def new_ticket():
    if request.method == 'POST':
        subject = request.form.get('subject', '').strip()
        body = request.form.get('body', '').strip()
        predicted_team = None
        prob = 0.0
        if ft_model is not None and clf is not None and body:
            try:
                predicted_team, prob, probs_dict = team(body, ft_model, clf, threshold=0.0)
            except Exception as e:
                print(f"[ERROR] Prediction failed: {e}")
                predicted_team, prob = None, 0.0
        
        print(f"[INFO] Predicted team: {predicted_team} with probability: {prob}")

        db.create_ticket(request.form['subject'], request.form['body'], predicted_team)
        return redirect(url_for('index'))
    return render_template('new_ticket.html')

@app.route('/ticket/<int:ticket_id>')
def view_ticket(ticket_id):
    ticket = db.get_ticket(ticket_id)
    if not ticket:
        return "Ticket not found", 404
    return render_template('view_ticket.html', ticket=ticket)

@app.route('/edit/<int:ticket_id>', methods=['GET', 'POST'])
def edit_ticket(ticket_id):
    ticket = db.get_ticket(ticket_id)
    if not ticket:
        return "Ticket not found", 404
    if request.method == 'POST':
        db.update_ticket(ticket_id, request.form['subject'], request.form['body'], request.form['team'])
        return redirect(url_for('index'))
    return render_template('edit_ticket.html', ticket=ticket)

@app.route('/delete/<int:ticket_id>', methods=['POST'])
def delete_ticket(ticket_id):
    db.delete_ticket(ticket_id)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
