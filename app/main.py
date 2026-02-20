from flask import Flask, render_template, request
import pandas as pd
import os
from sentiment import predict
from text_utils import clean_text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    df = pd.read_excel(file)

    comments = df.iloc[:,0].astype(str)

    results = []
    for comment in comments:
        clean = clean_text(comment)
        results.append(predict(clean))

    df["clasificacion"] = results

    output_path = "resultado.xlsx"
    df.to_excel(output_path, index=False)

    return render_template("result.html", table=df.head().to_html(), file_ready=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)