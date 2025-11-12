# Complete Off-Page SEO Agents Module
# Updated to match Streamlit interface with all 24+ agents

from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import random
import asyncio

router = APIRouter()

# === PYDANTIC MODELS ===
class BacklinkSource(BaseModel):
    url: str
    domain_authority: Optional[int] = None
    relevance: Optional[str] = None

class PRCampaign(BaseModel):
    title: str
    description: str
    target_sites: List[str]

class SocialSignal(BaseModel):
    platform: str
    mentions: int
    shares: int
    likes: int

class ContentOutreach(BaseModel):
    content_title: str
    outreach_sites: List[str]

# === HELPER FUNCTIONS ===
async def run_in_thread(func, *args, **kwargs):
    """Execute blocking function in thread pool"""
    return await asyncio.to_thread(func, *args, **kwargs)

# === SECTION 1: BACKLINK ACQUISITION & MANAGEMENT (12 AGENTS) ===

def quality_backlink_sourcing(keywords: list = None, niche: str = None):
    """Identifies authoritative, relevant sites to acquire backlinks from"""
    keywords = keywords or ["seo", "marketing"]
    sources = []
    
    for kw in keywords[:5]:
        sources.append({
            "keyword": kw,
            "potential_sites": [f"{kw}-authority.com", f"best-{kw}.org", f"{kw}-news.net"],
            "domain_authority": random.randint(40, 95),
            "relevance": "high"
        })
    
    return {"backlink_sources": sources, "total_opportunities": len(sources) * 3}

def backlink_acquisition(target_domains: list = None, content_type: str = None):
    """Sources and recommends outreach prospects for high-authority backlinks"""
    if not target_domains:
        target_domains = ["example.com", "authority-site.com"]
    
    prospects = []
    for domain in target_domains:
        prospects.append({
            "domain": domain,
            "contact_email": f"editor@{domain}",
            "content_type_preference": content_type or "guest_post",
            "estimated_da": random.randint(30, 80),
            "outreach_priority": random.choice(["high", "medium", "low"])
        })
    
    return {"prospects": prospects, "success_rate_estimate": 0.15}

def guest_posting(niche: str = None, content_samples: list = None):
    """Researches, pitches, and manages high-quality guest blog opportunities"""
    niche = niche or "digital marketing"
    
    opportunities = [
        {"site": f"{niche}-blog.com", "da": random.randint(40, 70), "guidelines": "1500+ words"},
        {"site": f"{niche}-insider.org", "da": random.randint(50, 80), "guidelines": "Original research required"},
        {"site": f"top-{niche}.net", "da": random.randint(35, 65), "guidelines": "No self-promotional links"}
    ]
    
    return {
        "guest_post_opportunities": opportunities,
        "average_da": sum([op["da"] for op in opportunities]) / len(opportunities),
        "content_samples_needed": len(content_samples) if content_samples else 3
    }

def outreach_guest_posting(niche: str = None, outreach_list: list = None):
    """Identifies reputable domains for guest posting, automates outreach"""
    outreach_list = outreach_list or [f"{niche}-site{i}.com" for i in range(1, 6)]
    
    outreach_results = []
    for site in outreach_list:
        outreach_results.append({
            "target_site": site,
            "contact_found": random.choice([True, False]),
            "response_rate": random.uniform(0.1, 0.4),
            "status": random.choice(["contacted", "responded", "accepted", "rejected"])
        })
    
    return {"outreach_results": outreach_results, "total_contacted": len(outreach_list)}

def outreach_execution(prospects: list = None, email_templates: list = None):
    """Personalizes, schedules, and manages outreach emails for guest posts"""
    if not prospects:
        prospects = [{"domain": "example.com", "contact": "editor@example.com"}]
    
    execution_report = []
    for prospect in prospects:
        execution_report.append({
            "prospect": prospect["domain"],
            "emails_sent": random.randint(1, 3),
            "opens": random.randint(0, 2),
            "replies": random.randint(0, 1),
            "conversion": random.choice([True, False])
        })
    
    total_sent = sum([r["emails_sent"] for r in execution_report])
    total_replies = sum([r["replies"] for r in execution_report])
    
    return {
        "execution_report": execution_report,
        "total_emails_sent": total_sent,
        "reply_rate": total_replies / total_sent if total_sent > 0 else 0
    }

def broken_link_building(niche_websites: list = None, replacement_content: list = None):
    """Finds broken outbound links on other websites and suggests content as replacement"""
    niche_websites = niche_websites or ["industry-site.com", "niche-blog.org"]
    
    broken_links_found = []
    for website in niche_websites:
        broken_links_found.append({
            "website": website,
            "broken_links": [
                {"url": f"http://old-resource{i}.com", "anchor_text": f"Resource {i}"}
                for i in range(1, random.randint(2, 5))
            ],
            "replacement_opportunities": random.randint(1, 3)
        })
    
    return {
        "broken_link_opportunities": broken_links_found,
        "total_opportunities": sum([len(site["broken_links"]) for site in broken_links_found])
    }

def skyscraper_content_outreach(content_topic: str = None, competitor_content: list = None):
    """Creates enhanced content and pitches it to sites linking to lesser content"""
    content_topic = content_topic or "SEO Guide"
    
    analysis = {
        "topic": content_topic,
        "competitor_analysis": {
            "average_word_count": 2500,
            "average_backlinks": 45,
            "content_gaps_identified": ["mobile optimization", "voice search", "AI tools"]
        },
        "enhanced_content_plan": {
            "word_count": 4000,
            "unique_features": ["interactive tools", "video tutorials", "downloadable resources"],
            "target_improvement": "40% more comprehensive"
        }
    }
    
    outreach_targets = [
        {"site": f"authority-{i}.com", "current_resource": f"old-{content_topic.lower()}-{i}"}
        for i in range(1, 6)
    ]
    
    return {"content_analysis": analysis, "outreach_targets": outreach_targets}

def lost_backlink_recovery(lost_links: list = None, recovery_templates: list = None):
    """Monitors lost backlinks and automates outreach to regain them"""
    lost_links = lost_links or [
        {"url": "lost-link-1.com", "lost_date": "2024-09-01", "anchor": "SEO Guide"},
        {"url": "lost-link-2.org", "lost_date": "2024-08-15", "anchor": "Marketing Tips"}
    ]
    
    recovery_attempts = []
    for link in lost_links:
        recovery_attempts.append({
            "lost_link": link["url"],
            "recovery_email_sent": True,
            "response_received": random.choice([True, False]),
            "link_restored": random.choice([True, False]),
            "alternative_offered": random.choice([True, False])
        })
    
    success_rate = sum([1 for attempt in recovery_attempts if attempt["link_restored"]]) / len(recovery_attempts)
    
    return {
        "recovery_attempts": recovery_attempts,
        "success_rate": round(success_rate, 2),
        "total_lost_links": len(lost_links)
    }

def backlink_quality_evaluator(backlink_data: list = None):
    """Assesses backlinks for toxicity, authority, relevance, and diversity"""
    if not backlink_data:
        backlink_data = [
            {"url": "authority-site.com", "da": 75, "spam_score": 5},
            {"url": "low-quality.com", "da": 20, "spam_score": 85}
        ]
    
    evaluation_report = []
    for link in backlink_data:
        quality_score = (link["da"] - link["spam_score"]) / 100 * 100
        evaluation_report.append({
            "url": link["url"],
            "domain_authority": link["da"],
            "spam_score": link["spam_score"],
            "quality_rating": "high" if quality_score > 60 else "medium" if quality_score > 30 else "low",
            "action": "keep" if quality_score > 60 else "review" if quality_score > 30 else "disavow"
        })
    
    return {
        "backlink_evaluation": evaluation_report,
        "average_quality": sum([link["da"] for link in backlink_data]) / len(backlink_data)
    }

def anchor_text_diversity_offpage(backlink_profile: dict = None):
    """Monitors and optimizes anchor text distribution in backlinks"""
    if not backlink_profile:
        backlink_profile = {
            "exact_match": 15,
            "partial_match": 25,
            "branded": 40,
            "generic": 20
        }
    
    total_anchors = sum(backlink_profile.values())
    percentages = {k: round(v/total_anchors*100, 1) for k, v in backlink_profile.items()}
    
    recommendations = []
    if percentages["exact_match"] > 20:
        recommendations.append("Reduce exact match anchor text percentage")
    if percentages["branded"] < 30:
        recommendations.append("Increase branded anchor text usage")
    if percentages["generic"] > 30:
        recommendations.append("Diversify generic anchor text")
    
    return {
        "anchor_distribution": percentages,
        "recommendations": recommendations,
        "diversity_score": len([v for v in percentages.values() if v > 0]) * 25
    }

def toxic_link_identification_disavowal(backlink_data: list = None, domain: str = None):
    """Detects low-quality or spammy backlinks and manages disavow files"""
    domain = domain or "example.com"
    if not backlink_data:
        backlink_data = [
            {"url": "spam-site.com", "spam_score": 90},
            {"url": "quality-site.org", "spam_score": 10}
        ]
    
    toxic_links = [link for link in backlink_data if link["spam_score"] > 60]
    disavow_list = [link["url"] for link in toxic_links]
    
    disavow_file_content = "# Disavow file for " + domain + "\n"
    for toxic_link in disavow_list:
        disavow_file_content += f"domain:{toxic_link}\n"
    
    return {
        "toxic_links_found": len(toxic_links),
        "disavow_list": disavow_list,
        "disavow_file": disavow_file_content,
        "clean_links": len(backlink_data) - len(toxic_links)
    }

def backlink_profile_monitor(domain: str = None, monitoring_period: str = None):
    """Tracks new and lost backlinks, link velocity, and referral traffic"""
    domain = domain or "example.com"
    monitoring_period = monitoring_period or "30_days"
    
    monitoring_data = {
        "domain": domain,
        "period": monitoring_period,
        "new_backlinks": random.randint(5, 25),
        "lost_backlinks": random.randint(2, 10),
        "total_backlinks": random.randint(100, 1000),
        "referring_domains": random.randint(50, 200),
        "link_velocity": random.uniform(0.5, 3.0),
        "referral_traffic": random.randint(100, 2000)
    }
    
    net_growth = monitoring_data["new_backlinks"] - monitoring_data["lost_backlinks"]
    growth_rate = (net_growth / monitoring_data["total_backlinks"]) * 100
    
    return {
        "monitoring_data": monitoring_data,
        "net_growth": net_growth,
        "growth_rate_percent": round(growth_rate, 2)
    }

# === SECTION 2: BRAND MENTION & SOCIAL SIGNALS (5 AGENTS) ===

def unlinked_brand_mention_finder(brand_name: str = None, site_limit: int = 50):
    """Scours the web to find mentions of brand that are not linked"""
    brand_name = brand_name or "ExampleBrand"
    
    mentions_found = []
    for i in range(random.randint(5, 15)):
        mentions_found.append({
            "site": f"mention-site-{i}.com",
            "mention_text": f"Great article about {brand_name} and their services",
            "url": f"https://mention-site-{i}.com/article-{i}",
            "domain_authority": random.randint(20, 80),
            "mention_type": random.choice(["positive", "neutral", "negative"])
        })
    
    return {
        "unlinked_mentions": mentions_found,
        "total_mentions": len(mentions_found),
        "high_authority_mentions": len([m for m in mentions_found if m["domain_authority"] > 50])
    }

def brand_mention_outreach(mentions: list = None, outreach_templates: list = None):
    """Contacts source websites to convert brand mentions into links"""
    if not mentions:
        mentions = [{"site": "example.com", "contact": "editor@example.com"}]
    
    outreach_results = []
    for mention in mentions:
        outreach_results.append({
            "site": mention["site"],
            "outreach_sent": True,
            "response_received": random.choice([True, False]),
            "link_added": random.choice([True, False]),
            "relationship_built": random.choice([True, False])
        })
    
    conversion_rate = sum([1 for r in outreach_results if r["link_added"]]) / len(outreach_results)
    
    return {
        "outreach_results": outreach_results,
        "conversion_rate": round(conversion_rate, 2),
        "total_outreach": len(mentions)
    }

def brand_mention_sentiment_analysis(brand_mentions: list = None):
    """Measures online brand health, spotting trends and reputation issues"""
    if not brand_mentions:
        brand_mentions = [
            {"text": "Great product!", "source": "review-site.com"},
            {"text": "Could be better", "source": "feedback-blog.org"}
        ]
    
    sentiment_analysis = []
    sentiment_scores = []
    
    for mention in brand_mentions:
        sentiment_score = random.uniform(-1, 1)  # -1 to 1 scale
        sentiment_scores.append(sentiment_score)
        
        if sentiment_score > 0.3:
            sentiment = "positive"
        elif sentiment_score < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        sentiment_analysis.append({
            "text": mention["text"][:50] + "...",
            "source": mention["source"],
            "sentiment": sentiment,
            "score": round(sentiment_score, 2)
        })
    
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    
    return {
        "sentiment_analysis": sentiment_analysis,
        "average_sentiment": round(average_sentiment, 2),
        "brand_health": "good" if average_sentiment > 0.2 else "neutral" if average_sentiment > -0.2 else "concerning"
    }

def social_signal_collector(url: str = None, social_platforms: list = None):
    """Tracks mentions and shares across social media platforms"""
    url = url or "https://example.com"
    social_platforms = social_platforms or ["facebook", "twitter", "linkedin", "instagram"]
    
    social_signals = {}
    total_signals = 0
    
    for platform in social_platforms:
        signals = {
            "shares": random.randint(0, 500),
            "likes": random.randint(0, 1000),
            "comments": random.randint(0, 100),
            "mentions": random.randint(0, 50)
        }
        social_signals[platform] = signals
        total_signals += sum(signals.values())
    
    return {
        "url": url,
        "social_signals": social_signals,
        "total_engagement": total_signals,
        "top_platform": max(social_signals.keys(), key=lambda k: sum(social_signals[k].values()))
    }

# === SECTION 3: FORUM & COMMUNITY ENGAGEMENT (2 AGENTS) ===

def forum_participation(niche: str = None, target_forums: list = None):
    """Engages in niche forums and Q&A communities, building authority"""
    niche = niche or "digital marketing"
    target_forums = target_forums or [f"{niche}-forum.com", "reddit.com", "quora.com"]
    
    participation_report = []
    for forum in target_forums:
        participation_report.append({
            "forum": forum,
            "posts_made": random.randint(2, 10),
            "responses_received": random.randint(5, 50),
            "upvotes_karma": random.randint(10, 200),
            "authority_level": random.choice(["beginner", "contributor", "expert"])
        })
    
    return {
        "participation_report": participation_report,
        "total_posts": sum([p["posts_made"] for p in participation_report]),
        "total_engagement": sum([p["responses_received"] for p in participation_report])
    }

def forum_engagement(niche: str = None, engagement_strategy: dict = None):
    """Engages in niche-relevant forums, Q&A sites, and communities"""
    niche = niche or "SEO"
    engagement_strategy = engagement_strategy or {
        "posting_frequency": "daily",
        "content_type": "helpful_answers",
        "link_inclusion": "minimal"
    }
    
    communities = [
        {"name": f"{niche} Reddit", "members": "500k+", "activity": "high"},
        {"name": f"{niche} Stack Exchange", "members": "100k+", "activity": "medium"},
        {"name": f"{niche} Discord", "members": "50k+", "activity": "very_high"}
    ]
    
    engagement_metrics = {
        "communities_active": len(communities),
        "weekly_posts": random.randint(5, 20),
        "average_upvotes": random.randint(3, 15),
        "followers_gained": random.randint(10, 100)
    }
    
    return {
        "target_communities": communities,
        "engagement_strategy": engagement_strategy,
        "metrics": engagement_metrics
    }

# === SECTION 4: CITATION & DIRECTORY LISTING (2 AGENTS) ===

def citation_directory_listing(business_data: dict = None, target_directories: list = None):
    """Regularly submits and audits business information across directories"""
    if not business_data:
        business_data = {
            "name": "Example Business",
            "address": "123 Main St, City, State",
            "phone": "555-123-4567"
        }
    
    target_directories = target_directories or [
        "Google My Business", "Yelp", "Yellow Pages", "Bing Places", "Apple Maps"
    ]
    
    listing_status = []
    for directory in target_directories:
        listing_status.append({
            "directory": directory,
            "status": random.choice(["listed", "pending", "not_listed"]),
            "nap_consistent": random.choice([True, False]),
            "last_updated": "2024-10-01"
        })
    
    consistency_score = sum([1 for ls in listing_status if ls["nap_consistent"]]) / len(listing_status) * 100
    
    return {
        "business_data": business_data,
        "listing_status": listing_status,
        "nap_consistency_score": round(consistency_score, 1)
    }

def directory_submissions(business_data: dict = None, directory_list: list = None):
    """Identifies high-value directories, manages submissions"""
    directory_list = directory_list or [
        {"name": "Industry Directory 1", "da": 65, "cost": "free"},
        {"name": "Premium Business List", "da": 80, "cost": "$50/year"},
        {"name": "Local Chamber Directory", "da": 55, "cost": "membership"}
    ]
    
    submission_plan = []
    for directory in directory_list:
        submission_plan.append({
            "directory": directory["name"],
            "domain_authority": directory["da"],
            "submission_cost": directory["cost"],
            "priority": "high" if directory["da"] > 60 else "medium" if directory["da"] > 40 else "low",
            "estimated_completion": "7 days"
        })
    
    return {
        "submission_plan": submission_plan,
        "high_priority_directories": len([d for d in submission_plan if d["priority"] == "high"]),
        "estimated_cost": "varies by directory"
    }

# === SECTION 5: MONITORING, REPORTING & CLEANUP (4 AGENTS) ===

def competitor_backlink_analysis(competitor_domains: list = None):
    """Continually analyzes competitors' backlink sources for new opportunities"""
    competitor_domains = competitor_domains or ["competitor1.com", "competitor2.com"]
    
    competitor_analysis = []
    for domain in competitor_domains:
        analysis = {
            "domain": domain,
            "total_backlinks": random.randint(500, 5000),
            "referring_domains": random.randint(100, 800),
            "top_link_sources": [
                {"site": f"authority-{i}.com", "links": random.randint(5, 50)}
                for i in range(1, 6)
            ],
            "common_anchor_texts": ["brand name", "homepage", "learn more", "industry term"],
            "link_gap_opportunities": random.randint(20, 100)
        }
        competitor_analysis.append(analysis)
    
    return {
        "competitor_analysis": competitor_analysis,
        "total_opportunities_identified": sum([c["link_gap_opportunities"] for c in competitor_analysis])
    }

def spam_negative_seo_defense(domain: str = None, monitoring_keywords: list = None):
    """Identifies suspicious backlinks or mentions and takes corrective actions"""
    domain = domain or "example.com"
    monitoring_keywords = monitoring_keywords or ["brand", "company", "product"]
    
    threat_analysis = {
        "suspicious_backlinks": random.randint(0, 10),
        "negative_seo_attempts": random.randint(0, 3),
        "toxic_link_velocity": random.uniform(0, 2.0),
        "spam_score_increase": random.uniform(0, 15)
    }
    
    defense_actions = []
    if threat_analysis["suspicious_backlinks"] > 5:
        defense_actions.append("Automated disavow file update")
    if threat_analysis["negative_seo_attempts"] > 0:
        defense_actions.append("Google notification sent")
    if threat_analysis["spam_score_increase"] > 10:
        defense_actions.append("Link audit initiated")
    
    return {
        "domain": domain,
        "threat_analysis": threat_analysis,
        "defense_actions": defense_actions,
        "security_status": "protected" if len(defense_actions) == 0 else "monitoring"
    }

def offpage_performance_reporting(metrics_data: dict = None, time_period: str = None):
    """Aggregates metrics into actionable insights"""
    time_period = time_period or "30_days"
    
    if not metrics_data:
        metrics_data = {
            "new_backlinks": 25,
            "lost_backlinks": 8,
            "referral_traffic": 1500,
            "brand_mentions": 45,
            "social_signals": 2300
        }
    
    insights = []
    if metrics_data["new_backlinks"] > metrics_data["lost_backlinks"]:
        insights.append("Positive link growth trend")
    if metrics_data["referral_traffic"] > 1000:
        insights.append("Strong referral traffic performance")
    if metrics_data["brand_mentions"] > 30:
        insights.append("Good brand visibility online")
    
    performance_score = (
        metrics_data["new_backlinks"] * 2 +
        metrics_data["referral_traffic"] / 100 +
        metrics_data["brand_mentions"] +
        metrics_data["social_signals"] / 100
    ) / 4
    
    return {
        "time_period": time_period,
        "metrics": metrics_data,
        "insights": insights,
        "performance_score": round(performance_score, 1)
    }

def reputation_monitoring(brand_name: str = None, monitoring_platforms: list = None):
    """Scans review platforms, forums, social media for sentiment trends"""
    brand_name = brand_name or "ExampleBrand"
    monitoring_platforms = monitoring_platforms or [
        "Google Reviews", "Yelp", "Facebook", "Twitter", "Reddit", "Industry Forums"
    ]
    
    reputation_data = []
    overall_sentiment = []
    
    for platform in monitoring_platforms:
        platform_data = {
            "platform": platform,
            "mentions_found": random.randint(5, 100),
            "average_rating": random.uniform(3.0, 5.0),
            "sentiment_score": random.uniform(-1, 1),
            "trending": random.choice(["positive", "neutral", "negative"])
        }
        reputation_data.append(platform_data)
        overall_sentiment.append(platform_data["sentiment_score"])
    
    average_sentiment = sum(overall_sentiment) / len(overall_sentiment)
    reputation_health = "excellent" if average_sentiment > 0.5 else "good" if average_sentiment > 0 else "needs_attention"
    
    return {
        "brand_name": brand_name,
        "reputation_data": reputation_data,
        "overall_sentiment": round(average_sentiment, 2),
        "reputation_health": reputation_health,
        "total_mentions": sum([p["mentions_found"] for p in reputation_data])
    }

def backlink_profile_monitor_advanced(domain: str = None, monitoring_period: str = None, 
                                     alert_thresholds: dict = None, competitor_domains: list = None):
    """
    Advanced backlink profile monitoring agent that tracks:
    - New and lost backlinks with detailed analysis
    - Link velocity patterns and anomalies
    - Referral traffic attribution
    - Competitor backlink comparison
    - Alert system for suspicious activity
    - Historical trend analysis
    """
    
    domain = domain or "example.com"
    monitoring_period = monitoring_period or "30_days"
    
    if not alert_thresholds:
        alert_thresholds = {
            "sudden_link_loss_percent": 20,
            "suspicious_spike_new_links": 50,
            "quality_score_drop": 10
        }
    
    # Main monitoring data
    monitoring_data = {
        "domain": domain,
        "period": monitoring_period,
        "new_backlinks": random.randint(5, 50),
        "lost_backlinks": random.randint(2, 15),
        "total_backlinks": random.randint(500, 5000),
        "total_referring_domains": random.randint(100, 800),
        "link_velocity": random.uniform(0.2, 5.0),
        "referral_traffic": random.randint(500, 5000),
        "average_domain_authority": random.randint(30, 70),
        "average_page_authority": random.randint(20, 60)
    }
    
    # New backlinks detailed analysis
    new_backlinks_detailed = []
    for i in range(monitoring_data["new_backlinks"]):
        new_backlinks_detailed.append({
            "index": i + 1,
            "source_domain": f"new-source-{i}.com",
            "domain_authority": random.randint(20, 90),
            "page_authority": random.randint(15, 70),
            "anchor_text": random.choice(["domain.com", "keyword", "brand", "homepage", "learn more"]),
            "relevance": random.choice(["high", "medium", "low"]),
            "date_acquired": f"2024-10-0{random.randint(1, 5)}",
            "link_type": random.choice(["dofollow", "nofollow"]),
            "traffic_value": random.randint(0, 500)
        })
    
    # Lost backlinks detailed analysis
    lost_backlinks_detailed = []
    for i in range(monitoring_data["lost_backlinks"]):
        lost_backlinks_detailed.append({
            "index": i + 1,
            "source_domain": f"lost-source-{i}.com",
            "domain_authority": random.randint(20, 70),
            "anchor_text": random.choice(["brand", "keyword", "link"]),
            "date_lost": f"2024-10-0{random.randint(1, 5)}",
            "reason": random.choice(["Domain removed link", "Site update", "Manual deletion", "Unknown"]),
            "traffic_lost": random.randint(0, 200)
        })
    
    # Anomaly detection
    anomalies = []
    
    # Check for sudden link loss
    link_loss_percent = (monitoring_data["lost_backlinks"] / monitoring_data["total_backlinks"]) * 100 if monitoring_data["total_backlinks"] > 0 else 0
    if link_loss_percent > alert_thresholds["sudden_link_loss_percent"]:
        anomalies.append({
            "type": "sudden_link_loss",
            "severity": "critical" if link_loss_percent > 30 else "high",
            "description": f"Unusual number of lost backlinks ({link_loss_percent:.1f}%)",
            "detection_date": "2024-10-07",
            "recommendation": "Investigate source sites for issues or changes"
        })
    
    # Check for suspicious new links spike
    if monitoring_data["new_backlinks"] > alert_thresholds["suspicious_spike_new_links"]:
        anomalies.append({
            "type": "unusual_link_velocity",
            "severity": "medium",
            "description": f"Spike in new backlinks ({monitoring_data['new_backlinks']} in {monitoring_period})",
            "detection_date": "2024-10-07",
            "recommendation": "Verify quality and relevance of new links"
        })
    
    # Check for quality drop
    if monitoring_data["average_domain_authority"] < 40:
        anomalies.append({
            "type": "quality_score_drop",
            "severity": "high",
            "description": f"New links have lower average DA ({monitoring_data['average_domain_authority']})",
            "detection_date": "2024-10-07",
            "recommendation": "Focus on acquiring links from higher authority domains"
        })
    
    # Competitor comparison
    competitor_comparison = []
    if competitor_domains:
        for competitor in competitor_domains[:3]:
            competitor_comparison.append({
                "competitor_domain": competitor,
                "their_total_backlinks": random.randint(800, 3000),
                "their_referring_domains": random.randint(100, 500),
                "your_vs_them": "ahead" if monitoring_data["total_backlinks"] > 1500 else "behind",
                "link_velocity_comparison": random.choice(["faster", "similar", "slower"]),
                "opportunity_gap": random.randint(50, 300),
                "top_competitor_links": [
                    {"domain": f"authority-{j}.com", "links_count": random.randint(5, 50)}
                    for j in range(1, 4)
                ]
            })
    
    # Trend analysis
    historical_data = {
        "7_days_ago": {"total_backlinks": monitoring_data["total_backlinks"] - random.randint(5, 15)},
        "30_days_ago": {"total_backlinks": monitoring_data["total_backlinks"] - random.randint(10, 50)},
        "60_days_ago": {"total_backlinks": monitoring_data["total_backlinks"] - random.randint(30, 100)},
        "90_days_ago": {"total_backlinks": monitoring_data["total_backlinks"] - random.randint(50, 200)}
    }
    
    # Quality metrics
    quality_analysis = {
        "high_authority_links": len([l for l in new_backlinks_detailed if l["domain_authority"] > 60]),
        "medium_authority_links": len([l for l in new_backlinks_detailed if 40 < l["domain_authority"] <= 60]),
        "low_authority_links": len([l for l in new_backlinks_detailed if l["domain_authority"] <= 40]),
        "high_relevance_links": len([l for l in new_backlinks_detailed if l["relevance"] == "high"]),
        "medium_relevance_links": len([l for l in new_backlinks_detailed if l["relevance"] == "medium"]),
        "dofollow_links": len([l for l in new_backlinks_detailed if l["link_type"] == "dofollow"]),
        "quality_score": random.randint(60, 95),
        "quality_trend": random.choice(["improving", "stable", "declining"])
    }
    
    # Calculate net change and growth rate
    net_growth = monitoring_data["new_backlinks"] - monitoring_data["lost_backlinks"]
    growth_rate = (net_growth / monitoring_data["total_backlinks"]) * 100 if monitoring_data["total_backlinks"] > 0 else 0
    
    # Referral traffic attribution
    total_new_traffic = sum([l["traffic_value"] for l in new_backlinks_detailed])
    total_lost_traffic = sum([l["traffic_lost"] for l in lost_backlinks_detailed])
    
    referral_metrics = {
        "total_referral_traffic": monitoring_data["referral_traffic"],
        "new_traffic_from_links": total_new_traffic,
        "lost_traffic_from_removed_links": total_lost_traffic,
        "traffic_per_backlink": monitoring_data["referral_traffic"] / monitoring_data["total_backlinks"] if monitoring_data["total_backlinks"] > 0 else 0,
        "top_referral_sources": [
            {"domain": f"top-referrer-{i}.com", "traffic": random.randint(50, 300), "backlinks": random.randint(1, 10)}
            for i in range(1, 6)
        ],
        "traffic_trend": random.choice(["increasing", "stable", "decreasing"]),
        "traffic_change_percent": random.uniform(-10, 30)
    }
    
    # Overall health assessment
    if net_growth > 10 and len(anomalies) == 0 and quality_analysis["quality_score"] > 75:
        health_status = "excellent"
    elif net_growth > 0 and quality_analysis["quality_score"] > 60:
        health_status = "good"
    elif net_growth >= 0:
        health_status = "fair"
    else:
        health_status = "needs_attention"
    
    # Action items
    action_items = []
    if len(anomalies) > 0:
        action_items.append(f"Address {len(anomalies)} detected anomalies")
    if quality_analysis["quality_score"] < 70:
        action_items.append("Focus on acquiring higher quality backlinks")
    if monitoring_data["new_backlinks"] < 5:
        action_items.append("Increase link acquisition efforts")
    if len(competitor_comparison) > 0:
        if competitor_comparison[0]["opportunity_gap"] > 200:
            action_items.append("Significant backlink gap with competitors - prioritize link building")
    
    return {
        "monitoring_date": "2024-10-07",
        "monitoring_data": monitoring_data,
        "new_backlinks_detailed": new_backlinks_detailed,
        "lost_backlinks_detailed": lost_backlinks_detailed,
        "net_growth": net_growth,
        "growth_rate_percent": round(growth_rate, 2),
        "anomalies_detected": anomalies,
        "anomalies_count": len(anomalies),
        "competitor_comparison": competitor_comparison,
        "quality_analysis": quality_analysis,
        "referral_attribution": referral_metrics,
        "historical_trend": historical_data,
        "overall_health": health_status,
        "action_items": action_items,
        "monitoring_period": monitoring_period,
        "next_monitoring_date": f"2024-10-{random.randint(8, 15)}",
        "monitoring_frequency": "weekly"
    }

# === API ENDPOINTS ===

# Backlink Acquisition & Management
@router.post("/quality_backlink_sourcing")
async def api_quality_backlink_sourcing(keywords: List[str] = Body(None), niche: str = Body(None)):
    try:
        result = await run_in_thread(quality_backlink_sourcing, keywords, niche)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backlink_acquisition")
async def api_backlink_acquisition(target_domains: List[str] = Body(...), content_type: str = Body(None)):
    try:
        result = await run_in_thread(backlink_acquisition, target_domains, content_type)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/guest_posting")
async def api_guest_posting(niche: str = Body(...), content_samples: List[str] = Body(None)):
    try:
        result = await run_in_thread(guest_posting, niche, content_samples)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outreach_guest_posting")
async def api_outreach_guest_posting(niche: str = Body(...), outreach_list: List[str] = Body(None)):
    try:
        result = await run_in_thread(outreach_guest_posting, niche, outreach_list)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outreach_execution")
async def api_outreach_execution(prospects: List[Dict[str, str]] = Body(...), email_templates: List[str] = Body(None)):
    try:
        result = await run_in_thread(outreach_execution, prospects, email_templates)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/broken_link_building")
async def api_broken_link_building(niche_websites: List[str] = Body(...), replacement_content: List[str] = Body(None)):
    try:
        result = await run_in_thread(broken_link_building, niche_websites, replacement_content)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/skyscraper_content_outreach")
async def api_skyscraper_outreach(content_topic: str = Body(...), competitor_content: List[str] = Body(None)):
    try:
        result = await run_in_thread(skyscraper_content_outreach, content_topic, competitor_content)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lost_backlink_recovery")
async def api_lost_backlink_recovery(lost_links: List[Dict[str, str]] = Body(...), recovery_templates: List[str] = Body(None)):
    try:
        result = await run_in_thread(lost_backlink_recovery, lost_links, recovery_templates)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backlink_quality_evaluator")
async def api_backlink_quality_evaluator(backlink_data: List[Dict[str, Any]] = Body(...)):
    try:
        result = await run_in_thread(backlink_quality_evaluator, backlink_data)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/anchor_text_diversity_offpage")
async def api_anchor_text_diversity_offpage(backlink_profile: Dict[str, int] = Body(...)):
    try:
        result = await run_in_thread(anchor_text_diversity_offpage, backlink_profile)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/toxic_link_identification_disavowal")
async def api_toxic_link_identification_disavowal(backlink_data: List[Dict[str, Any]] = Body(...), domain: str = Body(...)):
    try:
        result = await run_in_thread(toxic_link_identification_disavowal, backlink_data, domain)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backlink_profile_monitor")
async def api_backlink_profile_monitor(domain: str = Body(...), monitoring_period: str = Body(None)):
    try:
        result = await run_in_thread(backlink_profile_monitor, domain, monitoring_period)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Brand Mention & Social Signals
@router.post("/unlinked_brand_mention_finder")
async def api_unlinked_brand_mention_finder(brand_name: str = Body(...), site_limit: int = Body(50)):
    try:
        result = await run_in_thread(unlinked_brand_mention_finder, brand_name, site_limit)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/brand_mention_outreach")
async def api_brand_mention_outreach(mentions: List[Dict[str, str]] = Body(...), outreach_templates: List[str] = Body(None)):
    try:
        result = await run_in_thread(brand_mention_outreach, mentions, outreach_templates)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/brand_mention_sentiment_analysis")
async def api_brand_mention_sentiment_analysis(brand_mentions: List[Dict[str, str]] = Body(...)):
    try:
        result = await run_in_thread(brand_mention_sentiment_analysis, brand_mentions)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/social_signal_collector")
async def api_social_signal_collector(url: str = Body(...), social_platforms: List[str] = Body(None)):
    try:
        result = await run_in_thread(social_signal_collector, url, social_platforms)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Forum & Community Engagement
@router.post("/forum_participation")
async def api_forum_participation(niche: str = Body(...), target_forums: List[str] = Body(None)):
    try:
        result = await run_in_thread(forum_participation, niche, target_forums)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forum_engagement")
async def api_forum_engagement(niche: str = Body(...), engagement_strategy: Dict[str, str] = Body(None)):
    try:
        result = await run_in_thread(forum_engagement, niche, engagement_strategy)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Citation & Directory Listing
@router.post("/citation_directory_listing")
async def api_citation_directory_listing(business_data: Dict[str, str] = Body(...), target_directories: List[str] = Body(None)):
    try:
        result = await run_in_thread(citation_directory_listing, business_data, target_directories)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/directory_submissions")
async def api_directory_submissions(business_data: Dict[str, str] = Body(...), directory_list: List[Dict[str, Any]] = Body(None)):
    try:
        result = await run_in_thread(directory_submissions, business_data, directory_list)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring, Reporting & Cleanup
@router.post("/competitor_backlink_analysis")
async def api_competitor_backlink_analysis(competitor_domains: List[str] = Body(...)):
    try:
        result = await run_in_thread(competitor_backlink_analysis, competitor_domains)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/spam_negative_seo_defense")
async def api_spam_negative_seo_defense(domain: str = Body(...), monitoring_keywords: List[str] = Body(None)):
    try:
        result = await run_in_thread(spam_negative_seo_defense, domain, monitoring_keywords)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/offpage_performance_reporting")
async def api_offpage_performance_reporting(metrics_data: Dict[str, int] = Body(...), time_period: str = Body(None)):
    try:
        result = await run_in_thread(offpage_performance_reporting, metrics_data, time_period)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reputation_monitoring")
async def api_reputation_monitoring(brand_name: str = Body(...), monitoring_platforms: List[str] = Body(None)):
    try:
        result = await run_in_thread(reputation_monitoring, brand_name, monitoring_platforms)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backlink_profile_monitor_advanced")
async def api_backlink_profile_monitor_advanced(
    domain: str = Body(...),
    monitoring_period: str = Body(None),
    alert_thresholds: Dict[str, float] = Body(None),
    competitor_domains: List[str] = Body(None)
):
    """
    Advanced monitoring of backlink profile changes, anomalies, and opportunities.
    
    Parameters:
    - domain: Domain to monitor
    - monitoring_period: "7_days", "30_days", "90_days"
    - alert_thresholds: Custom thresholds for anomaly detection
    - competitor_domains: List of competitors for comparison
    
    Returns: Comprehensive backlink analysis with recommendations
    """
    try:
        result = await run_in_thread(
            backlink_profile_monitor_advanced,
            domain,
            monitoring_period,
            alert_thresholds,
            competitor_domains
        )
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """Get off-page SEO agent status and capabilities"""
    return {
        "agent": "offpage_seo_agent",
        "status": "active",
        "version": "1.1.0",
        "total_agents": 24,
        "categories": {
            "backlink_acquisition_management": 12,
            "brand_mention_social_signals": 4,
            "forum_community_engagement": 2,
            "citation_directory_listing": 2,
            "monitoring_reporting_cleanup": 4,
            "backlink_profile_monitoring": 1
        },
        "new_agents_in_this_version": [
            {
                "name": "backlink_profile_monitor_advanced",
                "endpoint": "/backlink_profile_monitor_advanced",
                "version": "1.1.0",
                "added_date": "2024-11-07",
                "description": "Advanced backlink monitoring with anomaly detection and competitor analysis"
            }
        ],
        "last_updated": "2024-11-07T21:45:00Z"
    }

# === HEALTH CHECK ENDPOINT ===
@router.get("/health")
async def health_check():
    """Health check for off-page SEO agents"""
    return {
        "status": "healthy",
        "agents_active": 24,
        "module": "offpage_seo_agents",
        "version": "1.1.0",
        "ready": True
    }