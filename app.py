from flask import Flask
from flask import render_template
from flask import request
from chatbot import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def chatbot_response():

    msg = request.args.get("msg")

    return get_response(msg)

if __name__ == "__main__":
    app.run(debug=True)