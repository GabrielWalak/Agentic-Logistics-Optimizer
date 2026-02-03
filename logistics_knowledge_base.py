"""
OLIST Logistics Knowledge Base
Enterprise-grade documentation for RAG system
"""
import os

LOGISTICS_DOCUMENTS = {
    "carrier_rules.txt": """
CARRIER RULES - OLIST LOGISTICS NETWORK

STANDARD SHIPPING:
- Maximum delivery time: 30 days
- Ideal weight: up to 2kg
- Cost: R$15-25 per shipment
- Coverage: All Brazil
- Late delivery penalty: 5% of order value
- Best for: Budget-conscious, non-urgent deliveries
- SLA: 95% on-time delivery rate
- Customer satisfaction: 3.8/5.0

PREMIUM EXPRESS:
- Maximum delivery time: 5-7 days
- Ideal weight: up to 5kg
- Cost: R$50-80 per shipment
- Coverage: SP, RJ, MG, PR (major hubs)
- Late delivery penalty: 2% of order value
- Best for: High-value items, time-sensitive orders
- SLA: 98% on-time delivery rate
- Customer satisfaction: 4.5/5.0

SEDEX LOGISTICS:
- Maximum delivery time: 10-14 days
- Ideal weight: up to 30kg
- Cost: R$30-45 per shipment
- Coverage: All Brazil
- Late delivery penalty: 3% of order value
- Best for: Medium-weight packages, balanced speed/cost
- SLA: 96% on-time delivery rate
- Customer satisfaction: 4.0/5.0

REGIONAL COURIER:
- Maximum delivery time: 1-3 days (local)
- Ideal weight: up to 1kg
- Cost: R$8-15 per shipment
- Coverage: Selected regions only
- Late delivery penalty: 1% of order value
- Best for: Last-mile local deliveries
- SLA: 99% on-time delivery rate
- Customer satisfaction: 4.7/5.0
    """,
    
    "distance_guidelines.txt": """
DELIVERY TIME GUIDELINES BY DISTANCE

LOCAL DELIVERY (0-100 km):
- Recommended carrier: LOCAL COURIER / STANDARD
- Average delivery: 1-3 days
- Risk level: MINIMAL ✓
- Recommendation: Standard Shipping suitable
- Weight impact: Minimal
- Peak season delay: +0-1 day
- Success rate: 98%
- Common issues: Traffic congestion in urban areas

REGIONAL DELIVERY (100-500 km):
- Recommended carrier: Regional Partner / SEDEX
- Average delivery: 3-7 days
- Risk level: MODERATE ⚠
- Recommendation: Premium Express for fragile items
- Weight impact: Moderate (+1 day per 5kg)
- Peak season delay: +1-2 days
- Success rate: 95%
- Common issues: Interstate border delays, weather conditions

INTERSTATE DELIVERY (500-1500 km):
- Recommended carrier: OLIST Main Network / Premium Express
- Average delivery: 7-15 days
- Risk level: HIGH 🔴
- Recommendation: Premium Express for high-value items (>R$500)
- Weight impact: Significant (+2-3 days per 5kg)
- Peak season delay: +2-3 days
- Success rate: 92%
- Common issues: Multiple handoffs, customs delays, infrastructure

LONG-DISTANCE DELIVERY (>1500 km):
- Recommended carrier: Premium Express (MANDATORY)
- Average delivery: 15-30 days
- Risk level: CRITICAL 🔴🔴
- Recommendation: ALWAYS use Premium or upgrade carrier
- Weight impact: Critical (+3-5 days per 5kg)
- Peak season delay: +3-5 days
- Success rate: 88%
- Common issues: Remote regions, limited carrier coverage
- Note: Consider split shipments for packages >10kg
    """,
    
    "weekend_holidays.txt": """
WEEKEND AND HOLIDAY IMPACT ON DELIVERY

WEEKEND PENALTY:
- Friday 18:00 - Sunday 23:59: +2-3 days delay expected
- Courier pickup: Suspended until Monday
- Processing: Resumes Monday 08:00 AM
- Monday impact: +1 additional day due to peak volume
- Recommendation: Auto-upgrade to Premium Express for weekend orders
- Historical data: 73% of weekend orders experience delay

BRAZILIAN HOLIDAYS (Impact on delivery):
- Christmas (Dec 25): +5-7 days | National shutdown
- New Year (Jan 1): +3-4 days | Reduced operations
- Carnival (February): +2-3 days (varies by year) | Regional impact
- Tiradentes Day (Apr 21): +1-2 days | National holiday
- Corpus Christi (variable): +1 day | Regional observance
- Independence Day (Sep 7): +1-2 days | National celebration
- Our Lady (Oct 12): +1 day | Religious observance
- Black Consciousness (Nov 20): +1 day | National holiday
- Labor Day (May 1): +1 day | National holiday

PEAK SEASON (November - December):
- General delay: +3-5 days across all carriers
- Volume increase: +200-300% vs normal operations
- Carrier recommendation: Upgrade minimum one tier
- Risk level: CRITICAL for Standard Shipping
- Black Friday (late Nov): +4-6 days
- Cyber Monday: +3-5 days
- Christmas rush (Dec 15-24): +7-10 days

STRATEGY:
- Weekend order? Force upgrade to Premium Express
- Holiday within 10 days? Add buffer +3 days to estimate
- November-December? Auto-upgrade all Standard → Premium
- Post-holiday period (Jan 2-15): Normal +1 day recovery delay
    """,
    
    "weight_volume_rules.txt": """
WEIGHT AND VOLUME IMPACT ON DELIVERY TIME

LIGHTWEIGHT PACKAGES (<500g):
- Delivery speed: FASTEST ⚡
- Carrier cost: MINIMUM
- Suitable carriers: Any
- Average delay vs baseline: -1 day
- Handling: Automated sorting possible
- Recommendation: Standard Shipping sufficient
- Packaging: Standard envelope or small box
- Damage rate: 0.5%

MEDIUM PACKAGES (500g - 2kg):
- Delivery speed: NORMAL
- Carrier cost: STANDARD
- Suitable carriers: Preferred Standard
- Average delay vs baseline: BASELINE
- Handling: Standard manual processing
- Recommendation: Standard Shipping OK
- Packaging: Standard box with minimal protection
- Damage rate: 1.2%

HEAVY PACKAGES (2kg - 5kg):
- Delivery speed: SLOWER (+1-2 days)
- Carrier cost: +30-50% surcharge
- Suitable carriers: SEDEX recommended
- Average delay vs baseline: +1-2 days
- Handling: Requires special handling
- Recommendation: SEDEX or Premium Express
- Packaging: Reinforced box with bubble wrap
- Damage rate: 2.1%

VERY HEAVY PACKAGES (5kg - 10kg):
- Delivery speed: SIGNIFICANTLY SLOWER (+3-5 days)
- Carrier cost: +100%+ surcharge
- Suitable carriers: MANDATORY Premium or SEDEX
- Average delay vs baseline: +3-5 days
- Handling: Requires special equipment (pallet jack)
- Recommendation: ALWAYS Premium/SEDEX
- Packaging: Heavy-duty box with corner protection
- Damage rate: 3.5%

OVERSIZED PACKAGES (>10kg):
- Delivery speed: EXTREMELY SLOW (+5-10 days)
- Carrier cost: +150%+ surcharge
- Suitable carriers: Premium Express ONLY
- Average delay vs baseline: +5-10 days
- Handling: Requires courier signature + special equipment
- Recommendation: Consider split shipments
- Packaging: Industrial-grade packaging required
- Damage rate: 5.2%
- Warning: May not be eligible for certain remote regions

VOLUME-TO-WEIGHT RATIO IMPACT:
- High volume, low weight: Standard processing
- Low volume, high weight: Slower delivery (density issues)
- Dimensional weight pricing applies when volume > 5000cm³
- Example: 1kg in 50L vs 1kg in 2L = same delivery speed
    """,
    
    "customer_recovery.txt": """
CUSTOMER RECOVERY AND RETENTION STRATEGY

VOUCHER CODE SYSTEM:

DELAY15 - 15% Discount
- Condition: Delay of 1-3 days vs promised delivery
- Customer impact: Moderate dissatisfaction (NPS: -10)
- Retention rate WITH voucher: 78%
- Retention rate WITHOUT: 45%
- Revenue impact: -15% order value, +33% retention
- Average LTV recovered: R$450 per customer
- Churn reduction: 33%

DELAY25 - 25% Discount
- Condition: Delay of 3-7 days vs promised delivery
- Customer impact: High dissatisfaction (NPS: -30)
- Retention rate WITH voucher: 82%
- Retention rate WITHOUT: 30%
- Revenue impact: -25% order value, +52% retention
- Average LTV recovered: R$650 per customer
- Churn reduction: 52%

DELAY50 - 50% Discount + FREE NEXT SHIPPING
- Condition: Delay >7 days vs promised delivery
- Customer impact: Critical dissatisfaction (NPS: -50)
- Retention rate WITH voucher: 85%
- Retention rate WITHOUT: 10%
- Revenue impact: -50% order value, +75% retention
- Average LTV recovered: R$800 per customer
- Churn reduction: 75%

EXPRESS_FREE - Free Upgraded Shipping
- Condition: Carrier fault (not customer/seller)
- Customer impact: Positive surprise (NPS: +20)
- Retention rate: 90%+
- Revenue impact: -R$50-80 shipping cost, excellent loyalty
- Average LTV recovered: R$1000 per customer
- Recommendation: Use proactively for VIP customers

COMMUNICATION PROTOCOL:
- Delay 1-3 days: Send DELAY15 within 24 hours of detection
- Delay 3-7 days: Send DELAY25 + tracking update + apology email
- Delay >7 days: Send DELAY50 + personal contact + resolution offer
- Carrier issue: Offer EXPRESS_FREE on next order + apology

TIMING STRATEGY:
- Send voucher BEFORE customer complains (proactive approach)
- Include real-time tracking update with apology
- Follow up when order arrives with satisfaction survey
- For VIP customers (>5 orders): upgrade voucher by one tier

ROI ANALYSIS:
- Cost of voucher: R$20-150 per incident
- Cost of lost customer: R$500-2000 LTV
- Recovery ROI: 300-500% for proactive voucher strategy
    """,
    
    "payment_lag_impact.txt": """
PAYMENT LAG IMPACT ON DELIVERY TIME

PAYMENT LAG = Days between order placement and shipment dispatch

IMMEDIATE PAYMENT (0-1 day):
- Shipping starts: Immediately after payment confirmed
- Risk level: MINIMAL ✓
- Processing delay: None
- Expected delay vs baseline: -5% (faster processing)
- Reliability: Very high (98%+ shipment rate)
- Fraud rate: 0.3%
- Recommendation: Standard Shipping OK
- Customer segment: Trusted repeat customers

NORMAL PAYMENT (1-3 days):
- Shipping starts: After standard verification (24-72h)
- Risk level: LOW ✓
- Processing delay: Standard queue
- Expected delay vs baseline: BASELINE
- Reliability: High (95%+ shipment rate)
- Fraud rate: 1.2%
- Recommendation: Standard Shipping suitable
- Customer segment: Regular customers

DELAYED PAYMENT (3-7 days):
- Shipping starts: After compliance review
- Risk level: MODERATE ⚠
- Processing delay: Manual verification needed (+2 days)
- Expected delay vs baseline: +2-3 days additional
- Reliability: Moderate (85-90% shipment rate)
- Fraud rate: 4.5%
- Chargeback risk: 2-5%
- Recommendation: Premium Express IF shipping proceeds
- Customer segment: New customers, payment issues

PAYMENT ISSUES (>7 days):
- Shipping starts: After extended risk assessment (if approved)
- Risk level: HIGH 🔴
- Processing delay: Extended compliance review (+5 days)
- Expected delay vs baseline: +5-10 days additional
- Reliability: Low (60-70% shipment rate)
- Fraud rate: 12%+
- Chargeback risk: 10-20%
- Cancellation probability: 30-40%
- Recommendation: Premium Express for approved orders only
- Customer segment: High-risk customers
- Note: Consider partial refund for customer satisfaction

STRATEGIC RECOMMENDATIONS:
- Payment lag >3 days? Auto-upgrade to Premium Express
- Payment lag >7 days? Require manager approval before shipping
- High-risk customer? Request prepaid Premium shipping
- First-time buyer with lag >5 days? Contact customer service

FINANCIAL IMPACT:
- Each day of payment lag = +0.5 days delivery delay
- Chargeback cost: R$50-200 per incident + order value
- Fraud prevention ROI: 400% (cost vs. prevented losses)
    """
}

def create_knowledge_base(base_path="./logistics_docs"):
    """
    Create knowledge base files for RAG system
    
    Args:
        base_path: Directory path for storing knowledge base documents
        
    Returns:
        List of created file paths
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    created_files = []
    for filename, content in LOGISTICS_DOCUMENTS.items():
        filepath = os.path.join(base_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(filepath)
    
    return created_files

def get_all_documents_text():
    """
    Get all knowledge base documents as concatenated text
    
    Returns:
        Dictionary with filename as key and content as value
    """
    return LOGISTICS_DOCUMENTS

if __name__ == "__main__":
    files = create_knowledge_base()
    print(f"✓ Knowledge base created: {len(files)} documents")
    for filepath in files:
        print(f"  - {filepath}")
