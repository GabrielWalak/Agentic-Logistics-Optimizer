"""
PydanticAI Multi-Agent System for OLIST Logistics
Enterprise-grade agent architecture with specialized responsibilities
Using direct Ollama API for reliability
"""
import os
import sys
import time
import re
import hashlib

# Fix encoding on Windows
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except (AttributeError, ValueError):
            pass  # Skip if stdout is already wrapped or unavailable (e.g., pytest)

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


# ===== GITHUB MODELS CLIENT =====

def _get_github_client() -> OpenAI:
    """Initialize OpenAI client configured for GitHub Models API"""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise ValueError("Missing GITHUB_TOKEN environment variable. Set it in .env or export GITHUB_TOKEN=...")
    base_url = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
    return OpenAI(base_url=base_url, api_key=token)


# ===== OLLAMA DIRECT API =====

class LLMError(Exception):
    """Raised when LLM call fails after all retries."""
    pass


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    stream: bool = False,
    max_retries: int = 3
) -> str:
    """LLM API call via GitHub Models using OpenAI SDK (backward-compatible name).
    
    Raises LLMError if all retries fail, instead of returning silent fallback JSON.
    """
    cache_key = rag_cache.make_key(system_prompt + user_prompt)
    cached = rag_cache.get(cache_key)
    if cached:
        if stream:
            print("[cached] ", end="", flush=True)
        return cached

    model_name = model or os.getenv("GITHUB_MODEL", "gpt-4o-mini")
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            client = _get_github_client()

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
                top_p=float(os.getenv("LLM_TOP_P", "0.9")),
                stream=stream,
            )

            if stream:
                full_response = ""
                for chunk in response:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_response += delta
                        print(delta, end="", flush=True)
                print()
                if not full_response.strip():
                    raise LLMError("LLM returned empty streaming response")
                rag_cache.set(cache_key, full_response)
                return full_response

            result = response.choices[0].message.content or ""
            if not result.strip():
                raise LLMError("LLM returned empty response")
            
            rag_cache.set(cache_key, result)
            return result

        except LLMError:
            raise  # Don't retry on empty response (model issue, not network)
        except Exception as e:
            last_error = e
            print(f"  ⚠ LLM attempt {attempt}/{max_retries} failed: {str(e)[:100]}")
            if attempt < max_retries:
                time.sleep(1.5 * attempt)  # Exponential backoff

    # All retries exhausted
    error_msg = f"GitHub Models API failed after {max_retries} attempts: {str(last_error)}"
    print(f"  ❌ {error_msg}")
    raise LLMError(error_msg)


def parse_json_response(response: str) -> Dict:
    """Extract JSON from LLM response (handles markdown code blocks).
    
    Raises ValueError if no valid JSON found.
    """
    try:
        # Remove markdown code blocks
        cleaned = response.replace('```json', '').replace('```', '').strip()
        
        # Find JSON object
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            parsed = json.loads(json_str)
            if parsed:
                return parsed
        
        # Fallback: try parsing entire response
        parsed = json.loads(cleaned)
        if parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    raise ValueError(f"Failed to parse JSON from LLM response: {response[:200]}")


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
    """Agent 1: Risk Assessment. Raises LLMError on failure."""
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
    if not factors:
        # LLM didn't provide factors — generate from scenario data
        factors = []
        if scenario.distance_km > 1500:
            factors.append(f"Long distance ({scenario.distance_km}km)")
        if scenario.weight_g > 3000:
            factors.append(f"Heavy weight ({scenario.weight_g}g)")
        if scenario.predicted_days - scenario.promised_days > 2:
            factors.append(f"Delivery delay ({scenario.predicted_days - scenario.promised_days:.1f} days)")
        if scenario.payment_lag_days > 3:
            factors.append(f"Payment lag ({scenario.payment_lag_days} days)")
        if scenario.is_weekend_order:
            factors.append("Weekend order")
        if not factors:
            factors = ["Standard delivery conditions"]
    
    analysis = result.get("analysis", "")
    if not analysis or analysis == "No analysis provided":
        # Generate meaningful analysis from data
        delay = scenario.predicted_days - scenario.promised_days
        analysis = (
            f"Delivery scenario analysis: {scenario.distance_km}km distance, "
            f"{scenario.weight_g}g weight, {delay:.1f} days potential delay. "
            f"Payment lag of {scenario.payment_lag_days} days. "
            f"Risk score {risk_score_value}/100 based on combined factors."
        )
    
    return {
        "risk_level": result.get("risk_level", "MODERATE"),
        "risk_score": risk_score_value,
        "primary_risk_factors": factors,
        "mitigation_priority": coerce_to_string(result.get("mitigation_priority"), "MEDIUM"),
        "analysis": analysis
    }


@traceable(name="carrier_optimization_agent")
def run_carrier_optimization(scenario: DeliveryScenario, risk: Dict) -> Dict:
    """Agent 2: Carrier Optimization. Raises LLMError on failure."""
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
    
    # Ensure ROI analysis is meaningful
    roi = result.get("roi_analysis", "")
    if not roi or len(roi) < 10:
        if should_upgrade:
            roi = (
                f"Upgrade cost R${cost_impact_value:.0f} justified by risk reduction. "
                f"Risk level {risk['risk_level']} ({risk['risk_score']}/100) requires faster carrier "
                f"to meet {scenario.promised_days}-day promise."
            )
        else:
            roi = "Standard carrier sufficient — no upgrade cost needed for current risk level."
    
    # Ensure upgrade_rationale is meaningful
    rationale = result.get("upgrade_rationale", "")
    if not rationale or len(rationale) < 10:
        if should_upgrade:
            rationale = (
                f"Risk level {risk['risk_level']} with {scenario.distance_km}km distance "
                f"requires carrier upgrade to ensure delivery within {scenario.promised_days} days."
            )
        else:
            rationale = "Current carrier meets delivery requirements at optimal cost."
    
    return {
        "recommended_carrier": result.get("recommended_carrier", "Premium Express"),
        "current_carrier": result.get("current_carrier", "Standard Shipping"),
        "should_upgrade": should_upgrade,
        "upgrade_rationale": rationale,
        "cost_impact": cost_impact_value,
        "roi_analysis": roi
    }


@traceable(name="recovery_strategy_agent")
def run_recovery_strategy(scenario: DeliveryScenario, risk: Dict) -> Dict:
    """Agent 3: Customer Recovery. Raises LLMError on failure."""
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
    
    # Ensure communication_template is meaningful
    template = result.get("communication_template", "")
    if not template or len(template) < 15:
        voucher = result.get("voucher_code")
        if voucher:
            template = (
                f"Subject: Update on your delivery | "
                f"We're proactively reaching out about a potential delay. "
                f"As a gesture of goodwill, here's your {voucher} code for {discount_value:.0f}% off."
            )
        else:
            template = "Subject: Your delivery is on track | We're monitoring your shipment and will notify you of any changes."
    
    # Ensure timing is meaningful
    timing = result.get("timing", "")
    if not timing or len(timing) < 5:
        if delay_days > 3:
            timing = "Day 1: Proactive notification with voucher"
        elif delay_days > 0:
            timing = "Day 1: Proactive notification, Day 3: Follow-up if delayed"
        else:
            timing = "Monitor only — no proactive outreach needed"
    
    return {
        "voucher_code": result.get("voucher_code"),
        "discount_percentage": discount_value,
        "communication_template": template,
        "timing": timing,
        "retention_probability": retention_value
    }


@traceable(name="orchestrator_agent")
def run_orchestrator(
    scenario: DeliveryScenario,
    risk: Dict,
    carrier: Dict,
    recovery: Dict
) -> Dict:
    """Agent 4: Decision Integration. Raises LLMError on failure."""
    user_prompt = f"""Integrate all agent recommendations:

Risk: {risk['risk_level']} ({risk['risk_score']}/100)
- Factors: {', '.join(risk.get('primary_risk_factors', [])[:3])}
Carrier: {carrier['recommended_carrier']} (Upgrade: {carrier['should_upgrade']}, Cost: R${carrier['cost_impact']})
Recovery: {recovery.get('voucher_code', 'None')} ({recovery['discount_percentage']}% discount)
- Retention probability: {recovery['retention_probability']}%

Scenario context:
- Distance: {scenario.distance_km}km, Weight: {scenario.weight_g}g
- Predicted: {scenario.predicted_days} days vs Promised: {scenario.promised_days} days

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
    if risk_level == 'CRITICAL':
        min_confidence = 70.0
    elif risk_level == 'HIGH':
        min_confidence = 75.0
    elif risk_level == 'MODERATE':
        min_confidence = 78.0
    elif risk_level == 'LOW':
        min_confidence = 85.0
    else:  # MINIMAL
        min_confidence = 90.0
    
    if confidence < min_confidence:
        confidence = min_confidence
    
    if abs(confidence - risk_score) < 5:
        confidence = min(confidence + 12, 95)
    
    confidence = max(70.0, min(confidence, 95.0))
    
    estimated_time = coerce_to_float(
        result.get("estimated_delivery_time"),
        scenario.predicted_days
    )
    
    # Ensure executive_summary is meaningful
    summary = result.get("executive_summary", "")
    if not summary or len(summary) < 20:
        delay = scenario.predicted_days - scenario.promised_days
        summary = (
            f"Risk level {risk_level} ({risk_score}/100). "
            f"{'Carrier upgrade to ' + carrier['recommended_carrier'] + ' recommended. ' if carrier['should_upgrade'] else 'Standard carrier sufficient. '}"
            f"{'Recovery voucher ' + str(recovery.get('voucher_code', '')) + ' activated. ' if recovery.get('voucher_code') else ''}"
            f"Estimated delivery: {estimated_time:.1f} days "
            f"({'on time' if delay <= 0 else f'{delay:.1f} days over promise'}). "
            f"Confidence: {confidence:.0f}%."
        )
    
    return {
        "executive_summary": summary,
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
    """Check if GitHub Models endpoint is reachable (kept name for compatibility).
    
    NOTE: On startup, we don't block waiting for API checks - just return False
    to avoid delaying server startup. Real LLM checks happen during requests.
    """
    # Don't perform actual LLM check on startup - it blocks the server
    # The system handles LLM unavailability gracefully during requests
    return False  # Assume not available on startup, will retry on first request


if __name__ == "__main__":
    print("PydanticAI Agent System - Direct Import Test\n")
    
    if not check_ollama_status():
        print("❌ Please start Ollama first")
        exit(1)

    # Test scenario
    scenario = DeliveryScenario(
        predicted_days=8.5,
        promised_days=7.0,
        distance_km=450,
        weight_g=1200,
        payment_lag_days=2,
        is_weekend_order=0,
        freight_value=45.00,
        rag_context="Standard carrier rules apply. Regional delivery expected."
    )

    result = run_multi_agent_analysis_parallel(scenario)
    print(result)


