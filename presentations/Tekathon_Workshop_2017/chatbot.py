from flask import Flask, render_template, request
from flask import jsonify
import chat_agent

app = Flask(__name__, static_url_path="/static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/message', methods=['POST'])
def reply():
    return jsonify( { 'text': chat_agent.duplicate_reply(request.form['msg']) } )

# start app
if __name__ == "__main__":
    app.run(port = 5000)