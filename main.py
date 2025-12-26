import os
import joblib
import pandas as pd
import numpy as np
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
from rich.markdown import Markdown

# --- Configuration ---
load_dotenv()
console = Console(force_terminal=True, color_system="truecolor")

# 1. INITIALIZATION (Safety against NameError)
model = None
client = None

# API Key Retrieval (Security first)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Fallback option - manually enter key if .env is missing
    api_key = "GROQ_API_KEY"

try:
    client = Groq(api_key=api_key)
except Exception as e:
    console.print(f"[bold red]❌ Groq Initialization Error:[/bold red] {e}")

# 2. MODEL LOADING
# Ensure the model file is in the same directory as main.py
model_filename = "xgboost_model.pkl"

try:
    model = joblib.load(model_filename)
    console.print(f"[bold green]✓ Model '{model_filename}' loaded successfully.[/bold green]")
except Exception as e:
    console.print(Panel(f"Model Loading Error: {e}\n\nPlease ensure 'xgboost' is installed (pip install xgboost)",
                        title="[bold red]CRITICAL ERROR[/bold red]", border_style="red"))


# 3. PREDICTION FUNCTION
def get_prediction(data_dict):
    if model is None:
        return 0.0

    features = [
        'product_weight_g', 'product_vol_cm3', 'distance_km', 'customer_lat', 'customer_lng',
        'seller_lat', 'seller_lng', 'payment_lag_days', 'is_weekend_order', 'freight_value',
        'purchase_month'
    ]

    # Maintain column order as expected by the trained XGBoost model
    df = pd.DataFrame([data_dict], columns=features)
    prediction = model.predict(df)
    return float(prediction[0])


# 4. AGENTIC WORKFLOW
def run_agentic_workflow(input_data, promised_days=7):
    if model is None or client is None:
        return 0.0, "System not properly initialized (Model or API key missing)."

    predicted_days = get_prediction(input_data)
    delay_risk = predicted_days - promised_days

    # AGENT PROMPT: Decision Intelligence & ROI Optimization
    prompt = f"""
    ROLE: Senior Logistics Strategy Consultant.
    SCENARIO: Analysis of a specific delivery route.
    - Predicted Delivery: {predicted_days:.1f} days
    - Promised Window: {promised_days} days
    - Payment Lag: {input_data['payment_lag_days']} days
    - Weekend Order: {'Yes' if input_data['is_weekend_order'] == 1 else 'No'}
    - Package Weight: {input_data['product_weight_g']}g
    - Distance: {input_data['distance_km']}km

    TASKS:
    1. RISK ASSESSMENT: Brief evaluation of the delivery delay risk.
    2. CUSTOMER RECOVERY: If predicted delay > 1 day vs promised, provide voucher code 'DELAY15'.
    3. CARRIER OPTIMIZATION: Suggest whether to stay with 'Standard Shipping' or switch to 'Premium Express' 
       to minimize contractual penalty costs and maximize ROI based on weight and distance.

    Format: Professional bullet points in English. Focus on ROI and strategic insights.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return predicted_days, chat_completion.choices[0].message.content
    except Exception as e:
        return predicted_days, f"AI Agent Communication Error: {e}"


# --- DATA SIMULATION ---
if __name__ == "__main__":
    # LIST OF SCENARIOS (High, Low, and Medium Risk)
    scenarios = [
        {
            "name": "🔴 CASE 1: High Risk (Distance & Weight)",
            "data": {
                'product_weight_g': 5000, 'product_vol_cm3': 15000, 'distance_km': 1200.0,
                'customer_lat': -23.5, 'customer_lng': -46.6, 'seller_lat': -15.7, 'seller_lng': -47.8,
                'payment_lag_days': 2, 'is_weekend_order': 1, 'freight_value': 85.00, 'purchase_month': 11
            }
        },
        {
            "name": "🟢 CASE 2: Low Risk (Local & Fast)",
            "data": {
                'product_weight_g': 500, 'product_vol_cm3': 1000, 'distance_km': 45.0,
                'customer_lat': -23.5, 'customer_lng': -46.6, 'seller_lat': -23.6, 'seller_lng': -46.7,
                'payment_lag_days': 0, 'is_weekend_order': 0, 'freight_value': 12.00, 'purchase_month': 5
            }
        },
        {
            "name": "🟡 CASE 3: Medium Risk (Payment Lag & Weekend)",
            "data": {
                'product_weight_g': 1200, 'product_vol_cm3': 5000, 'distance_km': 350.0,
                'customer_lat': -20.0, 'customer_lng': -40.0, 'seller_lat': -22.0, 'seller_lng': -43.0,
                'payment_lag_days': 4,
                'is_weekend_order': 1,
                'freight_value': 35.00, 'purchase_month': 10
            }
        }
    ]

    for scenario in scenarios:
        days, action_plan = run_agentic_workflow(scenario["data"])

        table = Table(title=f"[bold white]{scenario['name']}[/bold white]", box=None)  # box=None daje czystszy wygląd

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold magenta")

        table.add_row("Predicted Delivery", f"{days:.1f} Days")

        status_style = "bold red" if days > 7 else "bold green"
        status_text = "DELAY RISK" if days > 7 else "ON TIME"
        table.add_row("Status", f"[{status_style}]{status_text}[/{status_style}]")

        table.add_row("Decision Engine", "Llama-3.3 (70B) Agent")

        formatted_plan = Markdown(action_plan)

        console.print("\n")
        console.print(table)
        console.print(Panel(formatted_plan, title="Agent's Strategic Action Plan", border_style="blue", padding=(1, 2)))