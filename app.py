from flask import Flask, render_template, redirect, url_for
import subprocess
import threading

app = Flask(__name__)
game_running = False

def run_game():
    global game_running
    game_running = True
    subprocess.run(["python", "games.py"])
    game_running = False

@app.route('/')
def home():
    return render_template("index.html", game_running=game_running)

@app.route('/start-game')
def start_game():
    global game_running
    if not game_running:
        thread = threading.Thread(target=run_game)
        thread.start()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
