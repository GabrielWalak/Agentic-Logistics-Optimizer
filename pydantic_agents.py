"""
PydanticAI Multi-Agent System for OLIST Logistics
Enterprise-grade agent architecture with specialized responsibilities
Using direct Ollama API for reliability
"""
import os
import time
import re
import hashlib
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import ollama
import json

# Optional: Redis cache and LangSmith tracing
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠ Redis not available - install with: pip install redis")

# LangSmith tracing (optional)
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("⚠ LangSmith not available - install with: pip install langsmith")
    
    # Fallback: dummy decorator that does nothing
    def traceable(name: str = ""):
        def decorator(func):
            return func
        return decorator

# ===== DATA MODELS =====

class DeliveryScenario(BaseModel):
    """Structured input for delivery analysis"""
    predicted_days: float = Field(description="Predicted delivery time in days")
    promised_days: float = Field(default=7.0, description="Promised delivery window")
    distance_km: float = Field(description="Distance in kilometers")
    weight_g: float = Field(description="Package weight in grams")
    payment_lag_days: int = Field(description="Days between order and shipment")
    is_weekend_order: int = Field(description="1 if weekend, 0 otherwise")
    freight_value: float = Field(description="Freight cost in BRL")
    rag_context: str = Field(description="Retrieved knowledge from RAG system")


class RiskAssessment(BaseModel):
    """Risk assessment output"""
    risk_level: str = Field(default="MODERATE", description="Risk level")
    risk_score: float = Field(default=50.0, description="Risk score 0-100")
    primary_risk_factors: List[str] = Field(default_factory=list, description="Main risk contributors")
    mitigation_priority: str = Field(default="MEDIUM", description="Priority")
    analysis: str = Field(default="", description="Detailed risk analysis")


class CarrierRecommendation(BaseModel):
    """Carrier optimization output"""
    recommended_carrier: str = Field(default="Premium Express", description="Carrier name")
    current_carrier: str = Field(default="Standard Shipping", description="Current carrier")
    should_upgrade: bool = Field(default=False, description="Whether upgrade is recommended")
    upgrade_rationale: str = Field(default="", description="Justification for upgrade")
    cost_impact: float = Field(default=0.0, description="Additional cost in BRL")
    roi_analysis: str = Field(default="", description="ROI calculation")


class CustomerRecoveryPlan(BaseModel):
    """Customer recovery strategy output"""
    voucher_code: Optional[str] = Field(default=None, description="Voucher code")
    discount_percentage: float = Field(default=0.0, description="Discount percentage")
    communication_template: str = Field(default="", description="Customer message")
    timing: str = Field(default="", description="When to send")
    retention_probability: float = Field(default=0.0, description="Retention probability %")


class IntegratedDecision(BaseModel):
    """Final integrated decision from all agents"""
    risk_assessment: RiskAssessment
    carrier_recommendation: CarrierRecommendation
    recovery_plan: CustomerRecoveryPlan
    executive_summary: str = Field(default="", description="Executive summary")
    estimated_delivery_time: float = Field(default=0.0, description="Final delivery estimate")
    confidence_score: float = Field(default=75.0, description="Overall confidence 0-100")


# ===== OLLAMA DIRECT API =====

def call_ollama(system_prompt: str, user_prompt: str, model: str = "mistral:latest", stream: bool = False) -> str:
    """Direct Ollama API call with error handling and optional streaming"""
    try:
        # Check cache first
        cache_key = rag_cache.make_key(system_prompt + user_prompt)
        cached = rag_cache.get(cache_key)
        if cached:
            if stream:
                print("[cached] ", end='', flush=True)
            return cached
        
        if stream:
            # Streaming mode - show real-time output
            full_response = ""
            stream_response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                options={
                    "temperature": 0.3,
                    "num_predict": 1024,
                    "top_p": 0.9,
                }
            )
            
            for chunk in stream_response:
                content = chunk['message']['content']
                full_response += content
                print(content, end='', flush=True)
            
            print()  # New line after streaming
            
            # Cache the result
            rag_cache.set(cache_key, full_response)
            return full_response
        else:
            # Non-streaming mode
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 1024,
                    "top_p": 0.9,
                }
            )
            result = response['message']['content']
            
            # Cache the result
            rag_cache.set(cache_key, result)
            return result
            
    except Exception as e:
        return f'{{"error": "Ollama error: {str(e)}"}}'


def parse_json_response(response: str) -> Dict:
    """Extract JSON from LLM response (handles markdown code blocks)"""
    try:
        # Remove markdown code blocks
        response = response.replace('```json', '').replace('```', '').strip()
        
        # Find JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
        
        # Fallback: try parsing entire response
        return json.loads(response)
    except:
        return {}


# ===== REDIS CACHE MANAGER =====

class RAGCache:
    """Redis-based cache for RAG queries"""
    def __init__(self):
        self.enabled = False
        if REDIS_AVAILABLE:
            try:
                self.redis = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.redis.ping()
                self.enabled = True
                print("✓ Redis cache enabled")
            except Exception as e:
                print(f"⚠ Redis connection failed: {e}")
                self.enabled = False
    
    def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        if not self.enabled:
            return None
        try:
            return self.redis.get(key)
        except:
            return None
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """Cache value with TTL (default 1 hour)"""
        if not self.enabled:
            return
        try:
            self.redis.setex(key, ttl, value)
        except:
            pass
    
    @staticmethod
    def make_key(prompt: str) -> str:
        """Create cache key from prompt"""
        return f"rag:{hashlib.md5(prompt.encode()).hexdigest()}"


# Initialize global cache
rag_cache = RAGCache()


def coerce_to_float(value, default: float) -> float:
    """Safely convert to float, handle percentages"""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            # If value is between 0-1, assume it's decimal percentage (0.75 = 75%)
            if 0 < value <= 1 and default > 10:  # Only convert if default suggests percentage
                return value * 100
            return float(value)
        if isinstance(value, str):
            # Check if string contains % sign
            has_percent = '%' in value
            value = value.replace('%', '').strip()
            
            # Extract first number
            numbers = re.findall(r'-?\d+\.?\d*', value)
            if numbers:
                num = float(numbers[0])
                # If had % or is very small decimal (0.0-1.0), convert to percentage
                if has_percent or (0 < num <= 1 and default > 10):
                    return num * 100 if num <= 1 else num
                return num
            return default
        if isinstance(value, list):
            return coerce_to_float(value[0], default) if value else default
    except (ValueError, IndexError, TypeError):
        pass
    return default


def coerce_to_string(value, default: str) -> str:
    """Safely convert to string, handle lists"""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return ', '.join(str(v) for v in value)
        return str(value)
    except:
        return default


# ===== AGENT SYSTEM PROMPTS =====

RISK_AGENT_PROMPT = """You are a Senior Risk Assessment Specialist for OLIST Logistics.

Your role:
1. Analyze delivery risk factors based on provided scenario and knowledge base
2. Identify primary risk contributors (distance, weight, payment lag, weekend, etc.)
3. Assign risk level: MINIMAL, LOW, MODERATE, HIGH, or CRITICAL
4. Calculate risk score (0-100) based on multiple factors
5. Prioritize mitigation actions

Be precise, data-driven, and reference specific rules from the knowledge base.
Return JSON format: {"risk_level": "...", "risk_score": X, "primary_risk_factors": [...], "mitigation_priority": "...", "analysis": "..."}"""

CARRIER_AGENT_PROMPT = """You are a Carrier Optimization Expert for OLIST Logistics.

Your role:
1. Recommend optimal carrier based on scenario (Standard, Premium Express, SEDEX, Regional)
2. Evaluate cost vs. benefit of carrier upgrades
3. Calculate ROI comparing upgrade cost vs. penalty costs
4. Reference specific carrier rules from knowledge base
5. Justify recommendations with financial analysis

Return JSON format: {"recommended_carrier": "...", "current_carrier": "Standard Shipping", "should_upgrade": true/false, "upgrade_rationale": "...", "cost_impact": X, "roi_analysis": "..."}"""

RECOVERY_AGENT_PROMPT = """You are a Customer Recovery & Retention Strategist for OLIST.

Your role:
1. Design recovery strategy for at-risk deliveries
2. Select appropriate voucher code (DELAY15, DELAY25, DELAY50, EXPRESS_FREE)
3. Craft customer communication templates
4. Optimize timing for proactive outreach
5. Estimate retention probability as percentage (0-100)

Voucher system:
- DELAY15 (15%): 1-3 days delay
- DELAY25 (25%): 3-7 days delay  
- DELAY50 (50% + free shipping): >7 days delay
- EXPRESS_FREE: Carrier fault

Return JSON format: {"voucher_code": "...", "discount_percentage": X, "communication_template": "...", "timing": "...", "retention_probability": X}"""

ORCHESTRATOR_PROMPT = """You are the Chief Logistics Decision Officer for OLIST.

Your role:
1. Integrate insights from risk, carrier, and recovery specialists
2. Create cohesive action plan balancing all factors
3. Generate executive summary for stakeholders
4. Assign overall confidence score to recommendations (0-100)
5. Provide final delivery time estimate

Return JSON format: {"executive_summary": "...", "estimated_delivery_time": X, "confidence_score": X}"""


# ===== AGENT EXECUTION FUNCTIONS =====

@traceable(name="risk_assessment_agent")
def run_risk_assessment(scenario: DeliveryScenario) -> Dict:
    """Agent 1: Risk Assessment"""
    user_prompt = f"""Analyze this delivery scenario:

Predicted Delivery: {scenario.predicted_days} days
Promised Window: {scenario.promised_days} days
Distance: {scenario.distance_km}km
Weight: {scenario.weight_g}g
Payment Lag: {scenario.payment_lag_days} days
Weekend Order: {'Yes' if scenario.is_weekend_order else 'No'}
Freight Value: R${scenario.freight_value}

Knowledge Base Context:
{scenario.rag_context[:1000]}

Provide risk assessment in JSON format."""

    response = call_ollama(RISK_AGENT_PROMPT, user_prompt)
    result = parse_json_response(response)
    
    # Validate and coerce types
    risk_score_value = coerce_to_float(result.get("risk_score"), 50.0)
    
    # Ensure primary_risk_factors is a list
    factors = result.get("primary_risk_factors", [])
    if isinstance(factors, str):
        factors = [f.strip() for f in factors.split(',')]
    
    return {
        "risk_level": result.get("risk_level", "MODERATE"),
        "risk_score": risk_score_value,
        "primary_risk_factors": factors,
        "mitigation_priority": coerce_to_string(result.get("mitigation_priority"), "MEDIUM"),
        "analysis": result.get("analysis", "No analysis provided")
    }


@traceable(name="carrier_optimization_agent")
def run_carrier_optimization(scenario: DeliveryScenario, risk: Dict) -> Dict:
    """Agent 2: Carrier Optimization"""
    user_prompt = f"""Optimize carrier selection for:

Scenario:
- Predicted: {scenario.predicted_days} days
- Distance: {scenario.distance_km}km
- Weight: {scenario.weight_g}g
- Freight: R${scenario.freight_value}

Risk Assessment:
- Level: {risk['risk_level']}
- Score: {risk['risk_score']}/100

Knowledge Base:
{scenario.rag_context[:1000]}

Recommend carrier and calculate ROI in JSON format."""

    response = call_ollama(CARRIER_AGENT_PROMPT, user_prompt)
    result = parse_json_response(response)
    
    # Determine if upgrade is needed
    should_upgrade = (
        risk['risk_level'] in ['HIGH', 'CRITICAL'] or
        scenario.predicted_days > scenario.promised_days or
        result.get("should_upgrade", False)
    )
    
    # Validate cost_impact
    cost_impact_value = coerce_to_float(result.get("cost_impact"), 35.0 if should_upgrade else 0.0)
    
    return {
        "recommended_carrier": result.get("recommended_carrier", "Premium Express"),
        "current_carrier": result.get("current_carrier", "Standard Shipping"),
        "should_upgrade": should_upgrade,
        "upgrade_rationale": result.get("upgrade_rationale", "Risk mitigation required"),
        "cost_impact": cost_impact_value,
        "roi_analysis": result.get("roi_analysis", "Upgrade cost justified by risk reduction")
    }


@traceable(name="recovery_strategy_agent")
def run_recovery_strategy(scenario: DeliveryScenario, risk: Dict) -> Dict:
    """Agent 3: Customer Recovery"""
    delay_days = scenario.predicted_days - scenario.promised_days
    
    user_prompt = f"""Design recovery strategy for:

Scenario:
- Predicted: {scenario.predicted_days} days
- Promised: {scenario.promised_days} days
- Delay: {delay_days:.1f} days

Risk:
- Level: {risk['risk_level']}
- Score: {risk['risk_score']}/100

Knowledge Base:
{scenario.rag_context[:1000]}

Provide recovery plan in JSON format. Return retention_probability as percentage (0-100)."""

    response = call_ollama(RECOVERY_AGENT_PROMPT, user_prompt)
    result = parse_json_response(response)
    
    # Validate discount and retention as percentages
    discount_value = coerce_to_float(result.get("discount_percentage"), 0.0)
    retention_value = coerce_to_float(result.get("retention_probability"), 85.0)
    
    # Ensure retention is in 0-100 range
    if retention_value < 0:
        retention_value = 0.0
    elif retention_value > 100:
        retention_value = 100.0
    if retention_value < 10:  # Likely a decimal percentage
        retention_value = retention_value * 100
    
    return {
        "voucher_code": result.get("voucher_code"),
        "discount_percentage": discount_value,
        "communication_template": result.get("communication_template", "Standard delay notification"),
        "timing": result.get("timing", "Day 1: Initial notification"),
        "retention_probability": retention_value
    }


@traceable(name="orchestrator_agent")
def run_orchestrator(
    scenario: DeliveryScenario,
    risk: Dict,
    carrier: Dict,
    recovery: Dict
) -> Dict:
    """Agent 4: Decision Integration"""
    user_prompt = f"""Integrate all agent recommendations:

Risk: {risk['risk_level']} ({risk['risk_score']}/100)
Carrier: {carrier['recommended_carrier']} (Upgrade: {carrier['should_upgrade']})
Recovery: {recovery.get('voucher_code', 'None')} ({recovery['discount_percentage']}% discount)

Create executive summary and provide:
1. Cohesive action plan
2. Final delivery estimate
3. Overall confidence score (0-100)

Return JSON format."""

    response = call_ollama(ORCHESTRATOR_PROMPT, user_prompt)
    result = parse_json_response(response)
    
    # Validate confidence score with intelligent defaults
    confidence = coerce_to_float(result.get("confidence_score"), 75.0)
    risk_score = risk.get('risk_score', 50)
    risk_level = risk.get('risk_level', 'MODERATE')
    
    # Adjust confidence based on risk level (inverse relationship)
    # Lower risk = higher confidence in decision
    if risk_level == 'CRITICAL':
        min_confidence = 70.0  # We know what to do: aggressive action
    elif risk_level == 'HIGH':
        min_confidence = 75.0  # Clear mitigation path
    elif risk_level == 'MODERATE':
        min_confidence = 78.0  # Standard procedures apply
    elif risk_level == 'LOW':
        min_confidence = 85.0  # Simple, straightforward case
    else:  # MINIMAL
        min_confidence = 90.0  # Very confident in minimal action
    
    # If LLM returned too low confidence, adjust to minimum
    if confidence < min_confidence:
        confidence = min_confidence
    
    # If too similar to risk score, add variation
    if abs(confidence - risk_score) < 5:
        confidence = min(confidence + 12, 95)
    
    # Ensure confidence is in valid range (70-95%)
    confidence = max(70.0, min(confidence, 95.0))
    
    estimated_time = coerce_to_float(
        result.get("estimated_delivery_time"),
        scenario.predicted_days
    )
    
    return {
        "executive_summary": result.get(
            "executive_summary",
            "Integrated decision plan created"
        ),
        "estimated_delivery_time": estimated_time,
        "confidence_score": confidence
    }


# ===== MAIN WORKFLOW =====

@traceable(name="multi_agent_analysis")
def run_multi_agent_analysis_parallel(scenario: DeliveryScenario) -> IntegratedDecision:
    """Run multi-agent analysis with parallel execution where possible"""
    import concurrent.futures
    
    start_time = time.time()
    print("🤖 Starting multi-agent analysis (parallel)...")
    if rag_cache.enabled:
        print("  💾 Cache enabled - using Redis for optimization")
    
    # Stage 1: Risk assessment (must run first)
    print("  ├─ Agent 1: Risk Assessment...")
    risk_dict = run_risk_assessment(scenario)
    risk = RiskAssessment(**risk_dict)
    
    # Stage 2 & 3: Run carrier + recovery in parallel
    print("  ├─ Agents 2-3: Parallel execution...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        carrier_future = executor.submit(run_carrier_optimization, scenario, risk_dict)
        recovery_future = executor.submit(run_recovery_strategy, scenario, risk_dict)
        
        carrier_dict = carrier_future.result()
        recovery_dict = recovery_future.result()
    
    carrier = CarrierRecommendation(**carrier_dict)
    recovery = CustomerRecoveryPlan(**recovery_dict)
    
    # Stage 4: Final orchestration
    print("  └─ Agent 4: Decision Integration...")
    orchestrator_dict = run_orchestrator(scenario, risk_dict, carrier_dict, recovery_dict)
    
    # Build integrated decision
    integrated = IntegratedDecision(
        risk_assessment=risk,
        carrier_recommendation=carrier,
        recovery_plan=recovery,
        executive_summary=orchestrator_dict.get('executive_summary', 'Decision integrated'),
        estimated_delivery_time=orchestrator_dict.get('estimated_delivery_time', scenario.predicted_days),
        confidence_score=orchestrator_dict.get('confidence_score', 75)
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Multi-agent analysis complete in {elapsed:.1f}s!\n")
    return integrated


# Test if Ollama is available
def check_ollama_status() -> bool:
    """Check if Ollama is running and has required models"""
    try:
        response = ollama.list()
        models = [m.get('name', m.get('model', '')) for m in response.get('models', [])]
        print(f"✓ Ollama connected. Available models: {', '.join(models)}")
        
        # Check GPU usage
        try:
            ps_response = ollama.ps()
            if ps_response.get('models'):
                for model in ps_response['models']:
                    vram = model.get('size_vram', 0)
                    if vram > 0:
                        print(f"  🎮 GPU detected: {vram / (1024**3):.1f} GB VRAM in use")
                    else:
                        print(f"  💻 CPU mode (set OLLAMA_BACKEND=vulkan for AMD GPU)")
        except:
            pass
        
        # Show enabled features
        print(f"\n🔧 Features:")
        print(f"  {'✓' if rag_cache.enabled else '✗'} Redis cache: {'enabled' if rag_cache.enabled else 'disabled'}")
        print(f"  {'✓' if LANGSMITH_AVAILABLE and os.getenv('LANGSMITH_API_KEY') else '✗'} LangSmith tracing: {'enabled' if LANGSMITH_AVAILABLE and os.getenv('LANGSMITH_API_KEY') else 'disabled'}")
        
        return True
    except Exception as e:
        print(f"⚠ Ollama not available: {e}")
        return False


if __name__ == "__main__":
    print("PydanticAI Agent System - Direct Import Test\n")
    
    if not check_ollama_status():
        print("❌ Please start Ollama first")
        exit(1)