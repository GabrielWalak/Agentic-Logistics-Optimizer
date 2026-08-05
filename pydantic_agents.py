"""Multi-agent logistics workflow built with Pydantic contracts.

The module keeps orchestration explicit: specialized LLM calls produce typed
results and deterministic Python code validates business-critical values.
"""
import json
import os
import re
import hashlib
import sys
import time

# Fix encoding on Windows
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except (AttributeError, ValueError):
            pass  # Skip if stdout is already wrapped or unavailable (e.g., pytest)

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Type
from openai import OpenAI
from dotenv import load_dotenv

from carrier_tools import (
    CarrierQuote,
    get_all_carrier_quotes,
    select_carrier_quote,
)
from prompt_engineering import (
    CARRIER_AGENT_PROMPT,
    ORCHESTRATOR_PROMPT,
    RECOVERY_AGENT_PROMPT,
    RISK_AGENT_PROMPT,
)

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
    estimated_cost: float = Field(default=0.0, description="Verified quoted cost in BRL")
    estimated_transit_days: float = Field(default=0.0, description="Verified carrier transit estimate")
    quote_source: str = Field(default="", description="Source of carrier quote data")


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


# These contracts describe only values the LLM is allowed to propose. Verified
# quote values are deliberately absent because deterministic Python tools own
# operational price and transit-time data.
class CarrierAgentOutput(BaseModel):
    recommended_carrier: str
    current_carrier: str = "Standard Shipping"
    should_upgrade: bool
    upgrade_rationale: str
    roi_analysis: str


class OrchestratorOutput(BaseModel):
    executive_summary: str
    estimated_delivery_time: float
    confidence_score: float


# ===== PROVIDER-NEUTRAL LLM CLIENT =====

def get_llm_config() -> Dict[str, str]:
    """Resolve provider-neutral settings with temporary legacy compatibility."""
    return {
        "api_key": (
            os.getenv("LLM_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GITHUB_TOKEN", "").strip()
        ),
        "base_url": (
            os.getenv("LLM_BASE_URL", "").strip()
            or os.getenv("GITHUB_MODELS_BASE_URL", "").strip()
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        ),
        "model": (
            os.getenv("LLM_MODEL", "").strip()
            or os.getenv("GITHUB_MODEL", "").strip()
            or "gemini-3.6-flash"
        ),
    }


def _get_llm_client() -> OpenAI:
    """Initialize a bounded OpenAI-compatible client for the configured provider.

    SDK retries are disabled because ``call_ollama`` owns the retry policy. This
    prevents one logical attempt from expanding into nested, unobservable
    retries inside the client.
    """
    config = get_llm_config()
    if not config["api_key"]:
        raise ValueError(
            "Missing LLM_API_KEY environment variable. Set it to the API key "
            "issued by the configured LLM provider."
        )
    timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "20"))
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=timeout_seconds,
        max_retries=0,
    )


# ===== OLLAMA DIRECT API =====

class LLMError(Exception):
    """Raised when LLM call fails after all retries."""
    pass


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    stream: bool = False,
    max_retries: int = 3,
    response_model: Optional[Type[BaseModel]] = None,
) -> str:
    """Call an OpenAI-compatible LLM and optionally enforce a Pydantic schema.
    
    Raises LLMError if all retries fail, instead of returning silent fallback JSON.
    """
    config = get_llm_config()
    model_name = model or config["model"]
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    top_p = float(os.getenv("LLM_TOP_P", "0.9"))
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "low").strip()
    response_schema = (
        response_model.model_json_schema() if response_model is not None else None
    )

    if stream and response_model is not None:
        raise ValueError("Structured LLM responses cannot be streamed")

    # Model parameters are part of the key so configuration changes cannot
    # accidentally reuse an answer produced under different sampling settings.
    cache_material = json.dumps(
        {
            "model": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
            "base_url": config["base_url"],
            "response_schema": response_schema,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        sort_keys=True,
    )
    cache_key = rag_cache.make_key(cache_material)
    cached = rag_cache.get(cache_key)
    if cached:
        if stream:
            print("[cached] ", end="", flush=True)
        return cached

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            client = _get_llm_client()
            request = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
                "top_p": top_p,
            }

            # Gemini thinking models count internal reasoning against the output
            # limit. Keeping reasoning low leaves room for the required JSON.
            if "generativelanguage.googleapis.com" in config["base_url"]:
                request["reasoning_effort"] = reasoning_effort

            if response_model is not None:
                response = client.beta.chat.completions.parse(
                    **request,
                    response_format=response_model,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise LLMError("LLM returned no structured response")
                result = parsed.model_dump_json()
                rag_cache.set(cache_key, result)
                return result

            response = client.chat.completions.create(**request, stream=stream)

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
            # Authentication, invalid requests, and retired endpoints will not
            # recover during this request. Retrying them only adds latency.
            permanent_error = getattr(e, "status_code", None) in {
                400, 401, 403, 404, 410, 422
            }
            response_was_truncated = "length limit was reached" in str(e).lower()
            if permanent_error or response_was_truncated:
                break
            if attempt < max_retries:
                time.sleep(1.5 * attempt)  # Short incremental backoff

    # All retries exhausted
    error_msg = f"LLM API request failed: {str(last_error)}"
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
    """Best-effort Redis cache for LLM responses.

    The historical class name is kept for compatibility. Redis failures never
    fail an analysis; they only remove the cache optimization.
    """
    def __init__(self):
        self.enabled = False
        if REDIS_AVAILABLE:
            try:
                redis_url = os.getenv("REDIS_URL", "").strip()
                if redis_url:
                    self.redis = redis.Redis.from_url(
                        redis_url,
                        decode_responses=True,
                        socket_connect_timeout=2,
                    )
                else:
                    self.redis = redis.Redis(
                        host=os.getenv("REDIS_HOST", "localhost"),
                        port=int(os.getenv("REDIS_PORT", "6379")),
                        db=int(os.getenv("REDIS_DB", "0")),
                        decode_responses=True,
                        socket_connect_timeout=2,
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
        except Exception:
            return None
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """Cache value with TTL (default 1 hour)"""
        if not self.enabled:
            return
        try:
            self.redis.setex(key, ttl, value)
        except Exception:
            pass

    def is_healthy(self) -> bool:
        """Return the live cache status without leaking connection details."""
        if not self.enabled:
            return False
        try:
            return bool(self.redis.ping())
        except Exception:
            self.enabled = False
            return False
    
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
    except (ValueError, TypeError):
        return default


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

    response = call_ollama(
        RISK_AGENT_PROMPT,
        user_prompt,
        response_model=RiskAssessment,
    )
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
    """Agent 2: select a carrier using verified, typed quote data.

    The LLM explains the trade-off, while Python owns availability, price, and
    transit-time values. This boundary prevents generated financial data from
    being presented as an operational quote.
    """
    quotes: List[CarrierQuote] = get_all_carrier_quotes(
        distance_km=scenario.distance_km,
        weight_g=scenario.weight_g,
    )
    quote_payload = [quote.model_dump() for quote in quotes]

    user_prompt = f"""Optimize carrier selection for:

Scenario:
- Predicted: {scenario.predicted_days} days
- Promised: {scenario.promised_days} days
- Distance: {scenario.distance_km}km
- Weight: {scenario.weight_g}g
- Freight: R${scenario.freight_value}

Risk Assessment:
- Level: {risk['risk_level']}
- Score: {risk['risk_score']}/100

Verified carrier quotes produced by the carrier quote tool:
{json.dumps(quote_payload, ensure_ascii=False)}

Knowledge Base:
{scenario.rag_context[:1000]}

Select only an available carrier. Use the verified quote values and return the
required JSON object."""

    response = call_ollama(
        CARRIER_AGENT_PROMPT,
        user_prompt,
        response_model=CarrierAgentOutput,
    )
    result = parse_json_response(response)
    
    llm_upgrade = result.get("should_upgrade", False)
    if isinstance(llm_upgrade, str):
        llm_upgrade = llm_upgrade.strip().lower() in {"true", "yes", "1"}

    # Business rules have precedence over the model's upgrade preference.
    should_upgrade = (
        risk['risk_level'] in ['HIGH', 'CRITICAL'] or
        scenario.predicted_days > scenario.promised_days or
        bool(llm_upgrade)
    )

    selected_quote = select_carrier_quote(
        quotes=quotes,
        requested_carrier=coerce_to_string(
            result.get("recommended_carrier"),
            "Premium Express",
        ),
        should_upgrade=should_upgrade,
    )
    # Selecting any carrier other than the current one is operationally an
    # upgrade, even when the LLM returned an inconsistent boolean flag.
    should_upgrade = (
        should_upgrade
        or selected_quote.carrier != "Standard Shipping"
    )
    standard_quote = next(
        quote for quote in quotes if quote.carrier == "Standard Shipping"
    )
    cost_impact_value = round(
        max(0.0, selected_quote.estimated_cost - standard_quote.estimated_cost),
        2,
    )
    
    # Ensure ROI analysis is meaningful
    roi = result.get("roi_analysis", "")
    if not roi or len(roi) < 10:
        if should_upgrade:
            roi = (
                f"Verified upgrade cost impact is R${cost_impact_value:.2f}. "
                f"The selected quote estimates {selected_quote.estimated_transit_days:.1f} "
                f"transit days. Financial return cannot be fully quantified without "
                f"validated penalty and churn-cost data."
            )
        else:
            roi = (
                "The lowest-cost available quote meets the current requirements; "
                "no incremental carrier cost is required."
            )
    
    # Ensure upgrade_rationale is meaningful
    rationale = result.get("upgrade_rationale", "")
    if not rationale or len(rationale) < 10:
        if should_upgrade:
            rationale = (
                f"Risk level {risk['risk_level']} and the delivery window justify "
                f"the verified {selected_quote.carrier} quote."
            )
        else:
            rationale = "Current carrier meets delivery requirements at optimal cost."

    return {
        "recommended_carrier": selected_quote.carrier,
        "current_carrier": "Standard Shipping",
        "should_upgrade": should_upgrade,
        "upgrade_rationale": rationale,
        "cost_impact": cost_impact_value,
        "roi_analysis": roi,
        "estimated_cost": selected_quote.estimated_cost,
        "estimated_transit_days": selected_quote.estimated_transit_days,
        "quote_source": selected_quote.source,
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

    response = call_ollama(
        RECOVERY_AGENT_PROMPT,
        user_prompt,
        response_model=CustomerRecoveryPlan,
    )
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

    response = call_ollama(
        ORCHESTRATOR_PROMPT,
        user_prompt,
        response_model=OrchestratorOutput,
    )
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
    """Return whether an LLM provider is configured (legacy compatibility name).

    This is intentionally a local readiness check. Network reachability is
    verified by the authenticated debug probe or a real analysis request.
    """
    api_key = get_llm_config()["api_key"]
    return bool(api_key) and len(api_key) > 10


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


