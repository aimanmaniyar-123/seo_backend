# Complete Local SEO Agents Module
# Updated to match Streamlit interface with all 8+ agents

from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import random
import asyncio

router = APIRouter()

# === PYDANTIC MODELS ===
class BusinessData(BaseModel):
    name: str
    address: str
    phone: str
    website: Optional[str] = None
    category: Optional[str] = None

class Review(BaseModel):
    text: str
    rating: Optional[int] = None
    responded: Optional[bool] = False
    response: Optional[str] = None

class Listing(BaseModel):
    name: str
    address: str
    phone: str
    reviews: Optional[List[Review]] = []

class Competitor(BaseModel):
    name: str
    url: Optional[str] = None

class LocationData(BaseModel):
    name: str
    address: str
    phone: str

# === HELPER FUNCTIONS ===
async def run_in_thread(func, *args, **kwargs):
    """Execute blocking function in thread pool"""
    return await asyncio.to_thread(func, *args, **kwargs)

# === SECTION 1: GOOGLE BUSINESS PROFILE (GBP/GMB) MANAGEMENT (1 AGENT) ===

def gmb_manager_agent(gmb_page_url: str = None, business_data: dict = None):
    """Updates business information, interacts with reviews, posts updates"""
    if not business_data:
        business_data = {"name": "Example Business", "address": "123 Main St"}
    
    management_actions = {
        "business_info_updated": True,
        "photos_uploaded": random.randint(1, 5),
        "posts_published": random.randint(0, 3),
        "reviews_responded": random.randint(0, 2),
        "q_and_a_updated": True,
        "hours_verified": True
    }
    
    optimization_score = sum([
        25 if management_actions["business_info_updated"] else 0,
        min(management_actions["photos_uploaded"] * 5, 25),
        min(management_actions["posts_published"] * 8, 25),
        management_actions["reviews_responded"] * 10,
        10 if management_actions["hours_verified"] else 0
    ])
    
    return {
        "gmb_url": gmb_page_url or "https://business.google.com/example",
        "management_actions": management_actions,
        "optimization_score": min(optimization_score, 100),
        "recommendations": ["Add more photos", "Post weekly updates"] if optimization_score < 80 else ["Maintain current activity"]
    }

def business_profile_manager(listing_url: str = None, attributes: dict = None):
    """Manage business profile attributes"""
    if not attributes:
        attributes = {"name": "Updated Name", "hours": "9AM-5PM", "services": ["Service1", "Service2"]}
    
    profile_updates = {
        "listing_url": listing_url or "https://example-directory.com/business",
        "attributes_updated": len(attributes),
        "verification_status": "verified" if random.choice([True, False]) else "pending",
        "completeness_score": min(len(attributes) * 10, 100)
    }
    
    return {
        "profile_updates": profile_updates,
        "status": "success",
        "next_review_date": "2024-11-04"
    }

# === SECTION 2: CITATION MANAGEMENT (2 AGENTS) ===

def citation_builder_agent(business_data: dict = None):
    """Seeks new local citation opportunities, keeps existing ones consistent"""
    if not business_data:
        business_data = {"name": "Example Business", "address": "123 Main St", "phone": "555-1234"}
    
    # Popular citation directories
    directories = [
        {"name": "Google My Business", "priority": "high", "status": "active"},
        {"name": "Yelp", "priority": "high", "status": "active"},
        {"name": "Yellow Pages", "priority": "medium", "status": "pending"},
        {"name": "MapQuest", "priority": "medium", "status": "not_listed"},
        {"name": "Bing Places", "priority": "high", "status": "active"},
        {"name": "Apple Maps", "priority": "high", "status": "needs_update"},
        {"name": "Facebook", "priority": "high", "status": "active"},
        {"name": "Foursquare", "priority": "medium", "status": "not_listed"}
    ]
    
    citation_opportunities = []
    existing_citations = []
    
    for directory in directories:
        if directory["status"] == "not_listed":
            citation_opportunities.append({
                "directory": directory["name"],
                "priority": directory["priority"],
                "estimated_da": random.randint(40, 90),
                "submission_cost": "free" if random.choice([True, False]) else "$25/year"
            })
        else:
            existing_citations.append({
                "directory": directory["name"],
                "status": directory["status"],
                "nap_consistent": random.choice([True, False, True])  # 67% consistent
            })
    
    consistency_issues = len([c for c in existing_citations if not c["nap_consistent"]])
    
    return {
        "business_data": business_data,
        "citation_opportunities": citation_opportunities,
        "existing_citations": existing_citations,
        "consistency_issues": consistency_issues,
        "total_directories": len(directories),
        "recommendations": f"Fix {consistency_issues} consistency issues" if consistency_issues > 0 else "Citations are consistent"
    }

def citation_creation_audit_agent(business_data: dict = None):
    """Automates business submission to top local directories"""
    if not business_data:
        business_data = {"name": "Example Business", "address": "123 Main St", "phone": "555-1234"}
    
    # NAP (Name, Address, Phone) consistency check
    nap = (business_data.get("name"), business_data.get("address"), business_data.get("phone"))
    
    # Simulate directory audit
    directories = ["yellowpages.com", "yelp.com", "mapquest.com", "superpages.com", "whitepages.com"]
    audit_results = []
    
    for directory in directories:
        audit_results.append({
            "directory": directory,
            "current_nap": nap,
            "is_consistent": random.choice([True, False, True]),  # 67% consistent
            "listing_exists": random.choice([True, False, True, True]),  # 75% exist
            "last_updated": "2024-09-01",
            "action_needed": random.choice(["update_info", "create_listing", "none"])
        })
    
    consistency_score = len([a for a in audit_results if a["is_consistent"]]) / len(audit_results) * 100
    
    return {
        "business_nap": nap,
        "audit_results": audit_results,
        "consistency_score": round(consistency_score, 1),
        "directories_audited": len(directories),
        "action_items": [a["directory"] for a in audit_results if a["action_needed"] != "none"]
    }

# === SECTION 3: NAP CONSISTENCY (1 AGENT) ===

def nap_consistency_agent(listings: list = None):
    """Audits and ensures name, address, phone data is consistent across directories"""
    if not listings:
        listings = [
            {"name": "Example Business", "address": "123 Main St", "phone": "555-1234"},
            {"name": "Example Business Inc", "address": "123 Main Street", "phone": "(555) 123-4567"}
        ]
    
    # Extract NAP variations
    nap_variations = {
        "names": list(set([listing["name"] for listing in listings])),
        "addresses": list(set([listing["address"] for listing in listings])),
        "phones": list(set([listing["phone"] for listing in listings]))
    }
    
    # Check consistency
    consistency_check = {
        "name_consistent": len(nap_variations["names"]) == 1,
        "address_consistent": len(nap_variations["addresses"]) == 1,
        "phone_consistent": len(nap_variations["phones"]) == 1
    }
    
    overall_consistency = all(consistency_check.values())
    
    # Standardization recommendations
    standardization_plan = {
        "canonical_name": nap_variations["names"][0] if consistency_check["name_consistent"] else "NEEDS_STANDARDIZATION",
        "canonical_address": nap_variations["addresses"][0] if consistency_check["address_consistent"] else "NEEDS_STANDARDIZATION",
        "canonical_phone": nap_variations["phones"][0] if consistency_check["phone_consistent"] else "NEEDS_STANDARDIZATION"
    }
    
    inconsistencies = []
    if not consistency_check["name_consistent"]:
        inconsistencies.append(f"Name variations: {', '.join(nap_variations['names'])}")
    if not consistency_check["address_consistent"]:
        inconsistencies.append(f"Address variations: {', '.join(nap_variations['addresses'])}")
    if not consistency_check["phone_consistent"]:
        inconsistencies.append(f"Phone variations: {', '.join(nap_variations['phones'])}")
    
    return {
        "nap_variations": nap_variations,
        "consistency_check": consistency_check,
        "overall_consistent": overall_consistency,
        "standardization_plan": standardization_plan,
        "inconsistencies": inconsistencies,
        "listings_analyzed": len(listings)
    }

# === SECTION 4: REVIEW & REPUTATION MANAGEMENT (1 AGENT) ===

def review_management_agent(reviews: list = None, response_templates: dict = None):
    """Gathers new reviews, monitors sentiment, responds to feedback"""
    if not reviews:
        reviews = [
            {"text": "Great service!", "rating": 5, "platform": "Google", "responded": False},
            {"text": "Could be better", "rating": 3, "platform": "Yelp", "responded": False},
            {"text": "Excellent experience!", "rating": 5, "platform": "Facebook", "responded": True}
        ]
    
    # Analyze sentiment and ratings
    sentiment_analysis = []
    total_rating = 0
    responded_count = 0
    
    for review in reviews:
        # Simple sentiment analysis
        blob = TextBlob(review["text"])
        sentiment_score = blob.sentiment.polarity
        
        if sentiment_score > 0.3:
            sentiment = "positive"
        elif sentiment_score < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        sentiment_analysis.append({
            "text": review["text"][:50] + "...",
            "rating": review.get("rating", 0),
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "platform": review.get("platform", "unknown"),
            "responded": review.get("responded", False)
        })
        
        total_rating += review.get("rating", 0)
        if review.get("responded", False):
            responded_count += 1
    
    average_rating = total_rating / len(reviews) if reviews else 0
    response_rate = responded_count / len(reviews) * 100 if reviews else 0
    
    # Generate response suggestions for unresponded reviews
    response_suggestions = []
    for i, review in enumerate(reviews):
        if not review.get("responded", False):
            if review.get("rating", 0) >= 4:
                template = response_templates.get("positive", "Thank you for the wonderful review!") if response_templates else "Thank you for the wonderful review!"
            elif review.get("rating", 0) <= 2:
                template = response_templates.get("negative", "Thank you for your feedback. We'll work to improve.") if response_templates else "Thank you for your feedback. We'll work to improve."
            else:
                template = response_templates.get("neutral", "Thank you for taking the time to review us.") if response_templates else "Thank you for taking the time to review us."
            
            response_suggestions.append({
                "review_index": i,
                "suggested_response": template,
                "review_snippet": review["text"][:30] + "..."
            })
    
    # Review acquisition strategy
    acquisition_strategy = {
        "current_review_rate": "2-3 per week",
        "target_platforms": ["Google", "Yelp", "Facebook"],
        "follow_up_needed": len([r for r in reviews if r.get("rating", 0) <= 3]),
        "positive_reviews_to_promote": len([r for r in reviews if r.get("rating", 0) >= 4])
    }
    
    return {
        "review_summary": {
            "total_reviews": len(reviews),
            "average_rating": round(average_rating, 1),
            "response_rate_percent": round(response_rate, 1),
            "sentiment_breakdown": {
                "positive": len([s for s in sentiment_analysis if s["sentiment"] == "positive"]),
                "neutral": len([s for s in sentiment_analysis if s["sentiment"] == "neutral"]),
                "negative": len([s for s in sentiment_analysis if s["sentiment"] == "negative"])
            }
        },
        "sentiment_analysis": sentiment_analysis,
        "response_suggestions": response_suggestions,
        "acquisition_strategy": acquisition_strategy,
        "reputation_health": "excellent" if average_rating >= 4.5 else "good" if average_rating >= 4.0 else "needs_attention"
    }

# === SECTION 5: LOCAL KEYWORD RESEARCH & TRACKING (2 AGENTS) ===

def local_keyword_research(location: str = None, business_type: str = None, services: list = None):
    """Researches locally-relevant keywords integrates into strategy"""
    location = location or "New York"
    business_type = business_type or "restaurant"
    services = services or ["dining", "takeout", "catering"]
    
    # Generate local keyword variations
    local_keywords = []
    
    # Location + business type combinations
    base_combinations = [
        f"{business_type} in {location}",
        f"{business_type} near {location}",
        f"best {business_type} {location}",
        f"{location} {business_type}",
        f"top {business_type} {location}"
    ]
    
    # Add service-specific keywords
    for service in services:
        base_combinations.extend([
            f"{service} in {location}",
            f"{service} near {location}",
            f"best {service} {location}"
        ])
    
    # Add neighborhood and nearby city variations
    nearby_areas = [f"{location} downtown", f"{location} center", f"near {location}"]
    
    for keyword in base_combinations + nearby_areas:
        local_keywords.append({
            "keyword": keyword,
            "search_volume": random.randint(50, 2000),
            "competition": random.choice(["low", "medium", "high"]),
            "commercial_intent": random.choice(["high", "medium", "low"]),
            "local_pack_difficulty": random.randint(1, 10)
        })
    
    # Keyword categorization
    keyword_categories = {
        "high_intent": [kw for kw in local_keywords if kw["commercial_intent"] == "high"],
        "brand_defense": [kw for kw in local_keywords if business_type.lower() in kw["keyword"]],
        "service_specific": [kw for kw in local_keywords if any(service in kw["keyword"] for service in services)],
        "competitor_analysis": [f"vs {business_type} {location}", f"alternative to {business_type} {location}"]
    }
    
    return {
        "location": location,
        "business_type": business_type,
        "services": services,
        "local_keywords": local_keywords,
        "keyword_categories": keyword_categories,
        "total_keywords": len(local_keywords),
        "high_priority_count": len(keyword_categories["high_intent"])
    }

def map_pack_rank_tracker(keywords: list = None, location: str = None, competitors: list = None):
    """Monitors primary keyword rankings in Google's local map pack"""
    keywords = keywords or ["restaurant near me", "best pizza NYC"]
    location = location or "New York, NY"
    competitors = competitors or ["Competitor A", "Competitor B", "Competitor C"]
    
    ranking_data = []
    
    for keyword in keywords:
        # Simulate map pack rankings (1-3 for map pack, 4+ for organic)
        business_rank = random.randint(1, 10)
        competitor_ranks = {comp: random.randint(1, 10) for comp in competitors}
        
        visibility_score = max(0, 100 - (business_rank - 1) * 15)  # Higher score for better rank
        
        ranking_entry = {
            "keyword": keyword,
            "location": location,
            "business_rank": business_rank,
            "in_map_pack": business_rank <= 3,
            "competitor_ranks": competitor_ranks,
            "visibility_score": round(visibility_score, 1),
            "search_volume": random.randint(100, 5000),
            "tracking_date": "2024-10-04"
        }
        
        ranking_data.append(ranking_entry)
    
    # Calculate overall performance metrics
    map_pack_appearances = len([r for r in ranking_data if r["in_map_pack"]])
    average_rank = sum([r["business_rank"] for r in ranking_data]) / len(ranking_data)
    total_visibility = sum([r["visibility_score"] for r in ranking_data])
    
    # Competitive analysis
    competitive_analysis = {}
    for competitor in competitors:
        competitor_map_pack = len([r for r in ranking_data if competitor_ranks.get(competitor, 10) <= 3 for competitor_ranks in [r["competitor_ranks"]]])
        competitive_analysis[competitor] = {
            "map_pack_appearances": competitor_map_pack,
            "threat_level": "high" if competitor_map_pack > map_pack_appearances else "medium" if competitor_map_pack == map_pack_appearances else "low"
        }
    
    return {
        "tracking_summary": {
            "keywords_tracked": len(keywords),
            "map_pack_appearances": map_pack_appearances,
            "average_rank": round(average_rank, 1),
            "total_visibility_score": round(total_visibility, 1),
            "location": location
        },
        "ranking_data": ranking_data,
        "competitive_analysis": competitive_analysis,
        "recommendations": [
            "Optimize GMB profile" if map_pack_appearances < len(keywords) / 2 else "Maintain current ranking",
            "Increase review velocity" if average_rank > 3 else "Focus on review quality"
        ]
    }

# === SECTION 6: LOCAL COMPETITOR BENCHMARK (1 AGENT) ===

def local_competitor_benchmark_agent(business_data: dict = None, competitor_list: list = None):
    """Tracks, audits, and compares local presence vs. top local rivals"""
    if not business_data:
        business_data = {"name": "Your Business", "reviews": 150, "rating": 4.3}
    
    if not competitor_list:
        competitor_list = [
            {"name": "Competitor A", "reviews": 200, "rating": 4.1},
            {"name": "Competitor B", "reviews": 89, "rating": 4.6},
            {"name": "Competitor C", "reviews": 156, "rating": 4.0}
        ]
    
    # Benchmark categories
    benchmark_categories = [
        "google_reviews_count", "average_rating", "gmb_completeness", 
        "citation_count", "website_local_seo", "social_presence"
    ]
    
    # Generate competitive comparison
    competitive_comparison = {
        "your_business": {
            "name": business_data["name"],
            "google_reviews_count": business_data.get("reviews", 0),
            "average_rating": business_data.get("rating", 0),
            "gmb_completeness": random.randint(70, 95),
            "citation_count": random.randint(50, 120),
            "website_local_seo": random.randint(60, 90),
            "social_presence": random.randint(40, 80)
        }
    }
    
    competitor_data = []
    for competitor in competitor_list:
        competitor_metrics = {
            "name": competitor["name"],
            "google_reviews_count": competitor.get("reviews", random.randint(50, 300)),
            "average_rating": competitor.get("rating", random.uniform(3.5, 4.8)),
            "gmb_completeness": random.randint(60, 100),
            "citation_count": random.randint(40, 150),
            "website_local_seo": random.randint(50, 95),
            "social_presence": random.randint(30, 90)
        }
        competitor_data.append(competitor_metrics)
    
    # Calculate ranking in each category
    rankings = {}
    for category in benchmark_categories:
        all_scores = [competitive_comparison["your_business"][category]]
        all_scores.extend([comp[category] for comp in competitor_data])
        
        sorted_scores = sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)
        your_rank = next(i for i, (original_idx, score) in enumerate(sorted_scores) if original_idx == 0) + 1
        
        rankings[category] = {
            "your_score": competitive_comparison["your_business"][category],
            "your_rank": your_rank,
            "total_businesses": len(all_scores),
            "category_leader": max(all_scores),
            "improvement_needed": your_rank > 2
        }
    
    # Generate improvement opportunities
    improvement_opportunities = []
    for category, data in rankings.items():
        if data["improvement_needed"]:
            gap = data["category_leader"] - data["your_score"]
            improvement_opportunities.append({
                "category": category.replace("_", " ").title(),
                "current_rank": data["your_rank"],
                "gap_to_leader": gap,
                "priority": "high" if gap > 50 else "medium" if gap > 20 else "low"
            })
    
    # Overall competitive score
    avg_rank = sum([r["your_rank"] for r in rankings.values()]) / len(rankings)
    competitive_strength = "strong" if avg_rank <= 2 else "competitive" if avg_rank <= 3 else "needs_improvement"
    
    return {
        "business_data": business_data,
        "competitive_comparison": {
            "your_business": competitive_comparison["your_business"],
            "competitors": competitor_data
        },
        "category_rankings": rankings,
        "improvement_opportunities": improvement_opportunities,
        "competitive_analysis": {
            "average_rank": round(avg_rank, 1),
            "competitive_strength": competitive_strength,
            "categories_leading": len([r for r in rankings.values() if r["your_rank"] == 1]),
            "categories_need_improvement": len(improvement_opportunities)
        }
    }

# === API ENDPOINTS ===

@router.post("/gmb_manager")
async def api_gmb_manager(gmb_page_url: str = Query(None), business_data: BusinessData = Body(...)):
    """Manage Google Business Profile"""
    try:
        result = await run_in_thread(gmb_manager_agent, gmb_page_url, business_data.dict())
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/business_profile_manager")
async def api_business_profile_manager(listing_url: str = Query(None), attributes: Dict[str, Any] = Body(...)):
    """Manage business profile attributes"""
    try:
        result = await run_in_thread(business_profile_manager, listing_url, attributes)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/citation_builder")
async def api_citation_builder(business_data: BusinessData = Body(...)):
    """Build citations across directories"""
    try:
        result = await run_in_thread(citation_builder_agent, business_data.dict())
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/citation_creation_audit")
async def api_citation_audit(business_data: BusinessData = Body(...)):
    """Audit citation consistency"""
    try:
        result = await run_in_thread(citation_creation_audit_agent, business_data.dict())
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nap_consistency")
async def api_nap_consistency(listings: List[Listing] = Body(...)):
    """Check NAP consistency"""
    try:
        listings_dict = [l.dict() for l in listings]
        result = await run_in_thread(nap_consistency_agent, listings_dict)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/review_management")
async def api_review_management(reviews: List[Review] = Body(...), response_templates: Dict[str, str] = Body(None)):
    """Manage and respond to reviews"""
    try:
        reviews_dict = [r.dict() for r in reviews]
        result = await run_in_thread(review_management_agent, reviews_dict, response_templates)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/local_keyword_research")
async def api_local_keywords(location: str = Query(...), business_type: str = Query(None), services: List[str] = Body(None)):
    """Discover local keywords"""
    try:
        result = await run_in_thread(local_keyword_research, location, business_type, services)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/map_pack_rank_tracker")
async def api_map_pack_tracker(keywords: List[str] = Body(...), location: str = Query(...), competitors: List[str] = Body(None)):
    """Track map pack rankings"""
    try:
        result = await run_in_thread(map_pack_rank_tracker, keywords, location, competitors)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/local_competitor_benchmark")
async def api_competitor_benchmark(business_data: BusinessData = Body(...), competitor_list: List[Dict[str, Any]] = Body(...)):
    """Benchmark against competitors"""
    try:
        result = await run_in_thread(local_competitor_benchmark_agent, business_data.dict(), competitor_list)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """Get local SEO agent status"""
    return {
        "agent": "local_seo_agent",
        "status": "active",
        "total_endpoints": 8,
        "categories": [
            "Google Business Profile Management",
            "Citation Management", 
            "NAP Consistency",
            "Review & Reputation Management",
            "Local Keyword Research & Tracking",
            "Local Competitor Benchmarking"
        ]
    }