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
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Protocol,
    Type,
    TypeVar,
    cast,
)
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
_redis_module: Any = None
try:
    import redis as _redis_module
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠ Redis not available - install with: pip install redis")

# LangSmith tracing (optional)
P = ParamSpec("P")
R = TypeVar("R")


class _TraceableFactory(Protocol):
    """Subset of LangSmith's decorator API used by this module."""

    def __call__(
        self,
        *,
        name: str = "",
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def _noop_traceable(
    *,
    name: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Preserve decorated callables when LangSmith is unavailable."""
    del name

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return func

    return decorator


traceable: _TraceableFactory
try:
    from langsmith import traceable as _langsmith_traceable

    traceable = cast(_TraceableFactory, _langsmith_traceable)
    LANGSMITH_AVAILABLE = True
except ImportError:
    traceable = _noop_traceable
    LANGSMITH_AVAILABLE = False
    print("⚠ LangSmith not available - install with: pip install langsmith")


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


def _select_recovery_policy(
    delay_days: float,
) -> tuple[Optional[str], float]:
    """Return the voucher defined by the deterministic recovery policy."""
    if delay_days <= 0:
        return None, 0.0
    if delay_days <= 1:
        return "DELAY15", 15.0
    if delay_days <= 3:
        return "DELAY25", 25.0
    if delay_days <= 7:
        return "DELAY50", 50.0
    return "EXPRESS_FREE", 0.0


def calculate_risk_score(scenario: DeliveryScenario) -> float:
    """Calculate risk from the documented, auditable scoring rules."""
    score = 0.0
    if scenario.distance_km > 1500:
        score += 20
    if scenario.weight_g > 3000:
        score += 15
    if scenario.predicted_days - scenario.promised_days > 3:
        score += 25
    if scenario.payment_lag_days > 5:
        score += 10
    if scenario.is_weekend_order:
        score += 5
    return min(score, 100.0)


def risk_level_from_score(score: float) -> str:
    """Map a validated score to exactly one documented risk level."""
    if score <= 20:
        return "MINIMAL"
    if score <= 40:
        return "LOW"
    if score <= 60:
        return "MODERATE"
    if score <= 80:
        return "HIGH"
    return "CRITICAL"


def _risk_priority_from_level(risk_level: str) -> str:
    """Keep mitigation priority consistent with the validated risk level."""
    return {
        "MINIMAL": "LOW",
        "LOW": "MEDIUM",
        "MODERATE": "MEDIUM",
        "HIGH": "HIGH",
        "CRITICAL": "URGENT",
    }[risk_level]


def _validated_risk_factors(scenario: DeliveryScenario) -> List[str]:
    """Describe only the input conditions that contribute scoring points."""
    factors = []
    if scenario.distance_km > 1500:
        factors.append(f"Long distance ({scenario.distance_km:.0f} km, +20)")
    if scenario.weight_g > 3000:
        factors.append(f"Heavy package ({scenario.weight_g:.0f} g, +15)")
    delay_days = scenario.predicted_days - scenario.promised_days
    if delay_days > 3:
        factors.append(f"Predicted delay ({delay_days:.1f} days, +25)")
    if scenario.payment_lag_days > 5:
        factors.append(
            f"Payment lag ({scenario.payment_lag_days} days, +10)"
        )
    if scenario.is_weekend_order:
        factors.append("Weekend order (+5)")
    return factors or ["No documented scoring rule triggered"]


def _validated_risk_analysis(
    scenario: DeliveryScenario,
    risk_score: float,
    risk_level: str,
) -> str:
    """Build an explanation containing only validated shipment facts."""
    triggered_rules = []
    if scenario.distance_km > 1500:
        triggered_rules.append("distance over 1500 km (+20)")
    if scenario.weight_g > 3000:
        triggered_rules.append("weight over 3000 g (+15)")
    if scenario.predicted_days - scenario.promised_days > 3:
        triggered_rules.append("predicted delay over 3 days (+25)")
    if scenario.payment_lag_days > 5:
        triggered_rules.append("payment lag over 5 days (+10)")
    if scenario.is_weekend_order:
        triggered_rules.append("weekend order (+5)")

    rules_text = ", ".join(triggered_rules) or "no scoring rule triggered"
    return (
        f"Validated application rules produce {risk_score:.0f}/100, mapped to "
        f"{risk_level}. Triggered rules: {rules_text}. The ML model predicts "
        f"{scenario.predicted_days:.1f} days against a "
        f"{scenario.promised_days:.1f}-day promise. General knowledge-base "
        "statistics do not override these shipment-specific values."
    )


def _analysis_has_unsupported_claims(
    analysis: str,
    risk_score: float,
    risk_level: str,
) -> bool:
    """Detect common narrative contradictions before returning LLM text."""
    stated_levels = re.findall(
        r"\b(minimal|low|moderate|high|critical)\s+risk\b",
        analysis.lower(),
    )
    if any(level.upper() != risk_level for level in stated_levels):
        return True

    stated_scores = re.findall(r"\b(\d+(?:\.\d+)?)\s*/\s*100\b", analysis)
    if any(abs(float(score) - risk_score) > 0.01 for score in stated_scores):
        return True

    unsupported_statistic = re.search(
        r"\b\d+(?:\.\d+)?\s*(?:%|[-–]\s*\d+(?:\.\d+)?\s*days?\b)",
        analysis,
        flags=re.IGNORECASE,
    )
    return unsupported_statistic is not None


def build_deterministic_fallback_decision(
    scenario: DeliveryScenario,
) -> IntegratedDecision:
    """Build a transparent demo fallback from versioned business rules.

    The fallback is intentionally deterministic and contains no generated
    operational facts. It keeps the public portfolio usable during provider
    quota exhaustion while authenticated API calls continue to fail loudly.
    """
    risk_score = calculate_risk_score(scenario)
    delay_days = scenario.predicted_days - scenario.promised_days
    risk_factors = _validated_risk_factors(scenario)

    risk_level = risk_level_from_score(risk_score)
    risk = RiskAssessment(
        risk_level=risk_level,
        risk_score=risk_score,
        primary_risk_factors=risk_factors,
        mitigation_priority=_risk_priority_from_level(risk_level),
        analysis=_validated_risk_analysis(
            scenario,
            risk_score,
            risk_level,
        ),
    )

    quotes = get_all_carrier_quotes(
        distance_km=scenario.distance_km,
        weight_g=scenario.weight_g,
    )
    should_upgrade = risk_level in {"HIGH", "CRITICAL"} or delay_days > 0
    selected_quote = select_carrier_quote(
        quotes=quotes,
        requested_carrier=(
            "Premium Express" if should_upgrade else "Standard Shipping"
        ),
        should_upgrade=should_upgrade,
    )
    standard_quote = next(
        quote for quote in quotes if quote.carrier == "Standard Shipping"
    )
    cost_impact = round(
        max(0.0, selected_quote.estimated_cost - standard_quote.estimated_cost),
        2,
    )
    carrier = CarrierRecommendation(
        recommended_carrier=selected_quote.carrier,
        current_carrier="Standard Shipping",
        should_upgrade=(
            should_upgrade or selected_quote.carrier != "Standard Shipping"
        ),
        upgrade_rationale=(
            "Selected deterministically from available typed carrier quotes "
            "using the documented risk and delivery-window rules."
        ),
        cost_impact=cost_impact,
        roi_analysis=(
            f"Verified incremental quote cost is R${cost_impact:.2f}; financial "
            "return is not estimated without validated churn-cost data."
        ),
        estimated_cost=selected_quote.estimated_cost,
        estimated_transit_days=selected_quote.estimated_transit_days,
        quote_source=selected_quote.source,
    )

    voucher_code, discount = _select_recovery_policy(delay_days)

    recovery = CustomerRecoveryPlan(
        voucher_code=voucher_code,
        discount_percentage=discount,
        communication_template=(
            "Delivery status update generated from the documented recovery "
            "policy; notify the customer proactively when a delay is predicted."
        ),
        timing="Day 1 when a delay is detected" if delay_days > 0 else "Monitor only",
        retention_probability=max(50.0, 90.0 - max(delay_days, 0) * 4),
    )

    return IntegratedDecision(
        risk_assessment=risk,
        carrier_recommendation=carrier,
        recovery_plan=recovery,
        executive_summary=(
            f"Deterministic fallback: {risk_level} risk; use "
            f"{selected_quote.carrier} and apply the documented recovery policy."
        ),
        estimated_delivery_time=scenario.predicted_days,
        confidence_score=65.0,
    )


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

            if stream:
                # Literal stream flags let the SDK overloads expose the
                # correct response type to Pylance in each control-flow path.
                response_stream = client.chat.completions.create(
                    **request,
                    stream=True,
                )
                full_response = ""
                for chunk in response_stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_response += delta
                        print(delta, end="", flush=True)
                print()
                if not full_response.strip():
                    raise LLMError("LLM returned empty streaming response")
                rag_cache.set(cache_key, full_response)
                return full_response

            response = client.chat.completions.create(**request, stream=False)
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
                400, 401, 403, 404, 410, 422, 429
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

class _RedisClient(Protocol):
    """Minimal synchronous Redis interface used by the cache."""

    def ping(self) -> bool: ...

    def get(self, key: str) -> Optional[str]: ...

    def setex(self, key: str, ttl: int, value: str) -> Any: ...


class RAGCache:
    """Best-effort Redis cache for LLM responses.

    The historical class name is kept for compatibility. Redis failures never
    fail an analysis; they only remove the cache optimization.
    """
    def __init__(self) -> None:
        self.redis: Optional[_RedisClient] = None
        self.enabled = False
        if REDIS_AVAILABLE and _redis_module is not None:
            try:
                redis_url = os.getenv("REDIS_URL", "").strip()
                if redis_url:
                    self.redis = cast(
                        _RedisClient,
                        _redis_module.Redis.from_url(
                            redis_url,
                            decode_responses=True,
                            socket_connect_timeout=2,
                        ),
                    )
                else:
                    self.redis = cast(
                        _RedisClient,
                        _redis_module.Redis(
                            host=os.getenv("REDIS_HOST", "localhost"),
                            port=int(os.getenv("REDIS_PORT", "6379")),
                            db=int(os.getenv("REDIS_DB", "0")),
                            decode_responses=True,
                            socket_connect_timeout=2,
                        ),
                    )
                self.redis.ping()
                self.enabled = True
                print("✓ Redis cache enabled")
            except Exception as e:
                print(f"⚠ Redis connection failed: {e}")
                self.enabled = False
    
    def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        if not self.enabled or self.redis is None:
            return None
        try:
            return self.redis.get(key)
        except Exception:
            return None
    
    def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """Cache value with TTL (default 1 hour)"""
        if not self.enabled or self.redis is None:
            return
        try:
            self.redis.setex(key, ttl, value)
        except Exception:
            pass

    def is_healthy(self) -> bool:
        """Return the live cache status without leaking connection details."""
        if not self.enabled or self.redis is None:
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
    risk_score_value = calculate_risk_score(scenario)
    risk_level = risk_level_from_score(risk_score_value)
    user_prompt = f"""Analyze this delivery scenario:

Predicted Delivery: {scenario.predicted_days} days
Promised Window: {scenario.promised_days} days
Distance: {scenario.distance_km}km
Weight: {scenario.weight_g}g
Payment Lag: {scenario.payment_lag_days} days
Weekend Order: {'Yes' if scenario.is_weekend_order else 'No'}
Freight Value: R${scenario.freight_value}

Authoritative score calculated by application rules: {risk_score_value}/100
Authoritative risk level: {risk_level}

Knowledge Base Context:
{scenario.rag_context[:1000]}

Explain the validated result. Do not state a different score or level. The ML
prediction above is authoritative; retrieved general statistics must not
replace or contradict it. Provide the risk assessment in JSON format."""

    response = call_ollama(
        RISK_AGENT_PROMPT,
        user_prompt,
        response_model=RiskAssessment,
    )
    result = parse_json_response(response)
    
    factors = _validated_risk_factors(scenario)
    
    analysis = result.get("analysis", "")
    if (
        not analysis
        or analysis == "No analysis provided"
        or _analysis_has_unsupported_claims(
            analysis,
            risk_score_value,
            risk_level,
        )
    ):
        analysis = _validated_risk_analysis(
            scenario,
            risk_score_value,
            risk_level,
        )
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score_value,
        "primary_risk_factors": factors,
        "mitigation_priority": _risk_priority_from_level(risk_level),
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
    
    # Financial claims are never copied from model output. The model can choose
    # and explain a carrier, but only verified inputs may appear as numbers.
    if should_upgrade:
        roi = (
            f"Verified upgrade cost impact is R${cost_impact_value:.2f}. "
            f"The selected quote estimates "
            f"{selected_quote.estimated_transit_days:.1f} transit days. "
            "Financial return cannot be fully quantified without validated "
            "penalty and churn-cost data."
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
    
    # Voucher selection is a business rule, not a generative decision. Keeping
    # it in Python prevents a fluent response from bypassing the policy table.
    voucher_code, discount_value = _select_recovery_policy(delay_days)
    retention_value = coerce_to_float(result.get("retention_probability"), 85.0)
    
    # Ensure retention is in 0-100 range
    if retention_value < 0:
        retention_value = 0.0
    elif retention_value > 100:
        retention_value = 100.0
    if retention_value < 10:  # Likely a decimal percentage
        retention_value = retention_value * 100
    
    if voucher_code:
        benefit = (
            "free express delivery"
            if voucher_code == "EXPRESS_FREE"
            else f"{discount_value:.0f}% off"
        )
        template = (
            "Subject: Update on your delivery | We're proactively reaching "
            f"out about the predicted delay. Your {voucher_code} benefit "
            f"provides {benefit}."
        )
    else:
        template = (
            "Subject: Your delivery is on track | We're monitoring your "
            "shipment and will notify you if the forecast changes."
        )
    
    timing = (
        "Day 1: Proactive notification with recovery benefit"
        if voucher_code
        else "Monitor only - no proactive recovery needed"
    )

    return {
        "voucher_code": voucher_code,
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
- Estimated retention probability: {recovery['retention_probability']}%

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
    
    # Confidence describes evidence quality, not the inverse of operational
    # risk. Keep the model estimate bounded without artificially raising it.
    confidence = max(0.0, min(confidence, 90.0))
    
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


