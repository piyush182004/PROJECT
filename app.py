import os
from flask import Flask, request, render_template, jsonify
from models.predictor import predict_image
import ollama

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Store investment report globally to fine-tune chatbot responses
investment_report = ""

def generate_investment_analysis_gemma(class_name, user_data):
    """Use the Gemma-2B model to generate an investment analysis."""
    prompt = f"""
    Based on the given land classification '{class_name}' and the following user data:
    {user_data}

    Provide a detailed investment analysis, including:
    - Suitability for agricultural, industrial, or commercial use
    - Potential return on investment (ROI)
    - Risk factors and mitigation strategies
    - Market trends related to this land type
    - Suggested investment actions

    Keep the response structured and concise.
    """

    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

@app.route("/", methods=["GET", "POST"])
def home():
    """Render the main form with image upload & questions."""
    global investment_report  # Store the report for chatbot fine-tuning

    if request.method == "POST":
        # Handle image upload
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error="No file uploaded!")

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Predict land classification
        class_name, confidence = predict_image(file_path)

        # Collect user responses
        user_data = {key: request.form.get(key) for key in request.form.keys() if key != "file"}

        # Generate investment analysis using Gemma (2B)
        investment_report = generate_investment_analysis_gemma(class_name, user_data)

        return render_template(
            "result.html", 
            file_path=file_path, 
            class_name=class_name, 
            confidence=confidence,
            investment_advice=investment_report
        )
    
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot interactions using Llama3.2 (1B) fine-tuned with investment report."""
    global investment_report

    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Empty message!"}), 400
    
    # Provide investment report as context for chatbot responses
    chat_context = f"""
    Here is an investment analysis report based on land classification and user data:
    {investment_report}

    Now, answer the user's query based on this report.
    """

    response = ollama.chat(
        model="gemma2:2b", 
        messages=[
            {"role": "system", "content": "You are a financial assistant providing investment insights."},
            {"role": "system", "content": chat_context},
            {"role": "user", "content": user_message}
        ]
    )
    
    bot_response = response["message"]["content"]
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
