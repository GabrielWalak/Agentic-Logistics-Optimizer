"""
OLIST Logistics - Agentic AI System
Enterprise RAG architecture with ChromaDB + PydanticAI + Ollama Mistral
"""
import os
import joblib
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from dotenv import load_dotenv

# Import RAG and Agent systems
from chroma_db_manager import ChromaDBManager
from pydantic_agents import (
    DeliveryScenario, 
    run_multi_agent_analysis_parallel,
    check_ollama_status
)

# --- Configuration ---
load_dotenv()
console = Console(force_terminal=True, color_system="truecolor")

console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
console.print("[bold white]  OLIST AGENTIC AI - RAG-Powered Logistics System  [/bold white]")
console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n")

# 1. INITIALIZATION
model = None
chroma_manager = None

# Check Ollama status
console.print("🔍 [cyan]Checking Ollama status...[/cyan]")
if not check_ollama_status():
    console.print("[bold red]❌ Ollama not running![/bold red]")
    console.print("Please start Ollama and ensure 'mistral' model is available.")
    console.print("Run: [yellow]ollama pull mistral[/yellow]\n")
    exit(1)
console.print("[bold green]✓ Ollama connected[/bold green]\n")

# 2. XGBOOST MODEL LOADING
model_filename = "xgboost_model.pkl"
console.print(f"📦 [cyan]Loading XGBoost model: {model_filename}...[/cyan]")

try:
    model = joblib.load(model_filename)
    console.print(f"[bold green]✓ Model loaded successfully[/bold green]\n")
except Exception as e:
    console.print(Panel(
        f"Model Loading Error: {e}\n\nPlease ensure 'xgboost' is installed:\npip install xgboost",
        title="[bold red]CRITICAL ERROR[/bold red]", 
        border_style="red"
    ))
    exit(1)

# 3. CHROMADB INITIALIZATION
console.print("🗄️  [cyan]Initializing ChromaDB RAG system...[/cyan]")
try:
    chroma_manager = ChromaDBManager(
        persist_directory="./chroma_db",
        collection_name="olist_logistics_knowledge"
    )
    chroma_manager.index_knowledge_base(force_reindex=False)
    
    stats = chroma_manager.get_stats()
    console.print(f"[bold green]✓ ChromaDB ready[/bold green] ({stats['total_documents']} documents)\n")
except Exception as e:
    console.print(f"[bold yellow]⚠ ChromaDB initialization failed: {e}[/bold yellow]")
    console.print("[yellow]Continuing without RAG context...[/yellow]\n")
    chroma_manager = None



# 4. PREDICTION FUNCTION
def get_prediction(data_dict):
    """Get delivery time prediction from XGBoost model"""
    if model is None:
        return 0.0

    features = [
        'product_weight_g', 'product_vol_cm3', 'distance_km', 'customer_lat', 'customer_lng',
        'seller_lat', 'seller_lng', 'payment_lag_days', 'is_weekend_order', 'freight_value',
        'purchase_month'
    ]

    df = pd.DataFrame([data_dict], columns=features)
    prediction = model.predict(df)
    return float(prediction[0])


# 5. AGENTIC WORKFLOW WITH RAG
def run_agentic_workflow(input_data, promised_days=7):
    """
    Execute multi-agent analysis with RAG-enhanced context
    
    Args:
        input_data: Dictionary with delivery scenario data
        promised_days: Promised delivery window (default 7 days)
        
    Returns:
        Tuple of (predicted_days, analysis_result)
    """
    if model is None:
        return 0.0, "❌ XGBoost model not loaded"

    # Get ML prediction
    predicted_days = get_prediction(input_data)
    
    # Get RAG context from ChromaDB
    rag_context = ""
    if chroma_manager:
        try:
            rag_context = chroma_manager.get_relevant_context(
                predicted_days=predicted_days,
                promised_days=promised_days,
                input_data=input_data
            )
        except Exception as e:
            console.print(f"[yellow]⚠ RAG retrieval error: {e}[/yellow]")
            rag_context = "No context available"
    
    # Create scenario for PydanticAI agents
    scenario = DeliveryScenario(
        predicted_days=predicted_days,
        promised_days=promised_days,
        distance_km=input_data['distance_km'],
        weight_g=input_data['product_weight_g'],
        payment_lag_days=input_data['payment_lag_days'],
        is_weekend_order=input_data['is_weekend_order'],
        freight_value=input_data['freight_value'],
        rag_context=rag_context
    )
    
    # Run multi-agent analysis
    try:
        result = run_multi_agent_analysis_parallel(scenario)
        
        # Format output for display
        output = f"""## 🎯 INTEGRATED DECISION

### Risk Assessment
- **Risk Level**: {result.risk_assessment.risk_level}
- **Risk Score**: {result.risk_assessment.risk_score}/100
- **Primary Factors**: {', '.join(result.risk_assessment.primary_risk_factors)}
- **Analysis**: {result.risk_assessment.analysis}

### Carrier Recommendation
- **Recommended**: {result.carrier_recommendation.recommended_carrier}
- **Current**: {result.carrier_recommendation.current_carrier}
- **Should Upgrade**: {'✓ YES' if result.carrier_recommendation.should_upgrade else '✗ NO'}
- **Cost Impact**: R${result.carrier_recommendation.cost_impact:.2f}
- **ROI Analysis**: {result.carrier_recommendation.roi_analysis}

### Customer Recovery Plan
- **Voucher Code**: {result.recovery_plan.voucher_code or 'None'}
- **Discount**: {result.recovery_plan.discount_percentage}%
- **Timing**: {result.recovery_plan.timing}
- **Retention Probability**: {result.recovery_plan.retention_probability}%
- **Message**: {result.recovery_plan.communication_template}

### Executive Summary
{result.executive_summary}

**Confidence**: {result.confidence_score}/100
"""
        return predicted_days, output
        
    except Exception as e:
        console.print(f"[red]❌ Agent analysis error: {e}[/red]")
        return predicted_days, f"Error in multi-agent analysis: {e}"



# --- DATA SIMULATION ---
if __name__ == "__main__":
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
                'payment_lag_days': 4, 'is_weekend_order': 1, 'freight_value': 35.00, 'purchase_month': 10
            }
        }
    ]

    for scenario in scenarios:
        days, action_plan = run_agentic_workflow(scenario["data"])

        table = Table(title=f"[bold white]{scenario['name']}[/bold white]", box=None)

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold magenta")

        table.add_row("Predicted Delivery", f"{days:.1f} Days")

        status_style = "bold red" if days > 7 else "bold green"
        status_text = "DELAY RISK" if days > 7 else "ON TIME"
        table.add_row("Status", f"[{status_style}]{status_text}[/{status_style}]")

        table.add_row("Decision Engine", "🤖 Multi-Agent (Mistral 7B + RAG)")

        formatted_plan = Markdown(action_plan)

        console.print("\n")
        console.print(table)
        console.print(Panel(formatted_plan, title="🎯 Multi-Agent Strategic Analysis", border_style="blue", padding=(1, 2)))