from flask import Flask, render_template, request, redirect, url_for
import db
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

@app.route('/')
def index():
    tickets = db.get_all_tickets()
    return render_template('index.html', tickets=tickets)

@app.route('/new', methods=['GET', 'POST'])
def new_ticket():
    if request.method == 'POST':
        db.create_ticket(request.form['subject'], request.form['body'], request.form['team'])
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
