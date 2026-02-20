from flask import Flask, render_template, request, send_file
import pandas as pd
import os
from sentiment import predict
from text_utils import clean_text

app = Flask(__name__)

OUTPUT_FILE = "resultado.xlsx"

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

    # Guardar archivo
    df.to_excel(OUTPUT_FILE, index=False)

    # Conteo para gráfica
    counts = df["clasificacion"].value_counts().to_dict()

    normal = counts.get("normal", 0)
    regular = counts.get("regular", 0)
    toxico = counts.get("toxico", 0)

    return render_template(
        "result.html",
        normal=normal,
        regular=regular,
        toxico=toxico
    )

@app.route('/download')
def download():
    return send_file(OUTPUT_FILE, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)