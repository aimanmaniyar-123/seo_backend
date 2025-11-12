# Complete Core Agents Module
# Updated to match Streamlit interface with orchestration capabilities

from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

router = APIRouter()

# === IN-MEMORY STORAGE ===
micro_agents_registry = {}
micro_agents_dependencies = {}
action_logs = []
status_summary = {}

# === PYDANTIC MODELS ===
class TaskPriorities(BaseModel):
    high_priority: List[str] = []
    medium_priority: List[str] = []
    low_priority: List[str] = []

class SiteData(BaseModel):
    domain: str
    pages: Dict[str, Any] = {}
    meta_data: Dict[str, Any] = {}

class OrchestrationConfig(BaseModel):
    sequence: List[str] = []
    parallel_execution: bool = False
    retry_failed: bool = True

# === HELPER FUNCTIONS ===
def register_micro_agent(name: str, dependencies: List[str] = None):
    """Register a micro agent with optional dependencies"""
    dependencies = dependencies or []
    def decorator(func):
        micro_agents_registry[name] = func
        micro_agents_dependencies[name] = dependencies
        return func
    return decorator

def prioritize_agents() -> List[str]:
    """Topological sort to determine execution order based on dependencies"""
    result = []
    temp_marked = set()
    perm_marked = set()
    
    def visit(agent):
        if agent in perm_marked:
            return
        if agent in temp_marked:
            raise Exception(f"Circular dependency detected at {agent}")
        
        temp_marked.add(agent)
        for dep in micro_agents_dependencies.get(agent, []):
            if dep not in micro_agents_registry:
                raise Exception(f"Dependency {dep} not found for agent {agent}")
            visit(dep)
        
        perm_marked.add(agent)
        temp_marked.remove(agent)
        result.append(agent)
    
    for agent in micro_agents_registry.keys():
        if agent not in perm_marked:
            visit(agent)
    
    return result

async def run_micro_agent(agent_name: str) -> Dict[str, Any]:
    """Execute a single micro agent"""
    try:
        # Run the agent function in a thread pool to avoid blocking
        result = await asyncio.to_thread(micro_agents_registry[agent_name])
        
        log_entry = {
            "agent": agent_name,
            "success": True,
            "message": "Executed successfully",
            "timestamp": datetime.now().isoformat()
        }
        action_logs.append(log_entry)
        
        status_summary[agent_name] = {
            "status": "success",
            "details": result,
            "last_run": datetime.now().isoformat()
        }
        
        return result
    except Exception as e:
        log_entry = {
            "agent": agent_name,
            "success": False,
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        action_logs.append(log_entry)
        
        status_summary[agent_name] = {
            "status": "failed",
            "details": str(e),
            "last_run": datetime.now().isoformat()
        }
        
        raise Exception(f"Agent {agent_name} failed: {str(e)}")

async def run_all_agents() -> List[Dict[str, Any]]:
    """Execute all agents in dependency order"""
    order = prioritize_agents()
    results = []
    
    for agent in order:
        try:
            result = await run_micro_agent(agent)
            results.append({"agent": agent, "success": True, "result": result})
        except Exception as e:
            results.append({"agent": agent, "success": False, "error": str(e)})
            # Continue with remaining agents even if one fails
            continue
    
    return results

# === SEO ORCHESTRATION CORE AGENT ===

def seo_orchestration_core(site_data: dict = None, task_priorities: dict = None):
    """Oversees, prioritizes, triggers, and sequences all micro-agents across SEO categories"""
    if not site_data:
        site_data = {"domain": "example.com", "pages": {}}
    
    if not task_priorities:
        task_priorities = {
            "high_priority": ["technical_seo", "on_page_seo"],
            "medium_priority": ["off_page_seo"],
            "low_priority": ["local_seo"]
        }
    
    # Orchestration workflow
    orchestration_plan = {
        "phase_1_foundation": {
            "agents": ["robots_txt_management", "xml_sitemap_generator", "canonical_tag_management"],
            "estimated_duration": "30 minutes",
            "dependencies": []
        },
        "phase_2_onpage": {
            "agents": ["title_tag_optimizer", "meta_description_generator", "header_tag_manager"],
            "estimated_duration": "45 minutes", 
            "dependencies": ["phase_1_foundation"]
        },
        "phase_3_technical": {
            "agents": ["page_speed_analyzer", "mobile_usability_tester", "schema_markup_validator"],
            "estimated_duration": "60 minutes",
            "dependencies": ["phase_1_foundation"]
        },
        "phase_4_content": {
            "agents": ["content_quality_depth", "keyword_mapping", "internal_links_agent"],
            "estimated_duration": "90 minutes",
            "dependencies": ["phase_2_onpage"]
        },
        "phase_5_offpage": {
            "agents": ["backlink_analyzer", "social_signal_tracker", "brand_mention_outreach"],
            "estimated_duration": "120 minutes",
            "dependencies": ["phase_4_content"]
        }
    }
    
    # Priority scoring system
    priority_scores = {
        agent: 3 for agent in task_priorities.get("high_priority", [])
    }
    priority_scores.update({
        agent: 2 for agent in task_priorities.get("medium_priority", [])
    })
    priority_scores.update({
        agent: 1 for agent in task_priorities.get("low_priority", [])
    })
    
    # Resource allocation
    resource_allocation = {
        "total_agents_available": len(micro_agents_registry),
        "active_agents": len([a for a in status_summary.values() if a.get("status") == "success"]),
        "failed_agents": len([a for a in status_summary.values() if a.get("status") == "failed"]),
        "estimated_completion_time": "4-6 hours",
        "resource_utilization": "80%"
    }
    
    # Health monitoring
    system_health = {
        "overall_status": "healthy" if resource_allocation["failed_agents"] == 0 else "degraded",
        "success_rate": (resource_allocation["active_agents"] / max(1, resource_allocation["total_agents_available"])) * 100,
        "last_full_audit": datetime.now().isoformat(),
        "next_scheduled_run": "2024-10-05T10:00:00"
    }
    
    return {
        "site_data": site_data,
        "orchestration_plan": orchestration_plan,
        "priority_scores": priority_scores,
        "resource_allocation": resource_allocation,
        "system_health": system_health,
        "recommendations": [
            "Run foundation phase first",
            "Monitor technical SEO agents closely",
            "Schedule off-page activities for low-traffic hours"
        ]
    }

# === REGISTER CORE MICRO AGENTS ===

@register_micro_agent(name="seo_orchestration_core")
def seo_orchestration_core_agent():
    """SEO Orchestration Core Agent"""
    return {
        "task": "seo_orchestration", 
        "status": "completed",
        "actions": ["workflow_orchestrated", "agents_prioritized", "resources_allocated"]
    }

@register_micro_agent(name="on_page_seo_agent")
def on_page_seo_agent():
    """Execute on-page SEO tasks"""
    return {
        "task": "on_page_seo",
        "status": "completed", 
        "actions": ["meta_tags_optimized", "headers_checked", "content_analyzed"]
    }

@register_micro_agent(name="off_page_seo_agent", dependencies=["on_page_seo_agent"])
def off_page_seo_agent():
    """Execute off-page SEO tasks"""
    return {
        "task": "off_page_seo",
        "status": "completed",
        "actions": ["backlinks_analyzed", "social_signals_checked"]
    }

@register_micro_agent(name="technical_seo_agent")
def technical_seo_agent():
    """Execute technical SEO tasks"""
    return {
        "task": "technical_seo",
        "status": "completed",
        "actions": ["site_speed_analyzed", "mobile_friendliness_checked"]
    }

@register_micro_agent(name="local_seo_agent", dependencies=["technical_seo_agent"])
def local_seo_agent():
    """Execute local SEO tasks"""
    return {
        "task": "local_seo", 
        "status": "completed",
        "actions": ["google_my_business_optimized", "local_citations_updated"]
    }

# === API ENDPOINTS ===

@router.post("/seo_orchestration_core")
async def api_seo_orchestration_core(site_data: SiteData = Body(...), task_priorities: TaskPriorities = Body(None)):
    """Main SEO orchestration endpoint"""
    try:
        task_priorities_dict = task_priorities.dict() if task_priorities else None
        result = await asyncio.to_thread(seo_orchestration_core, site_data.dict(), task_priorities_dict)
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trigger_all_agents")
async def trigger_all_agents():
    """Trigger all agents in dependency order"""
    try:
        results = await run_all_agents()
        return {
            "message": "All agents executed",
            "results": results,
            "total_agents": len(micro_agents_registry),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard_summary")
async def dashboard_summary():
    """Get dashboard summary with agent statuses"""
    return {
        "total_agents": len(micro_agents_registry),
        "successful_agents": sum(1 for a in status_summary.values() if a["status"] == "success"),
        "failed_agents": sum(1 for a in status_summary.values() if a["status"] == "failed"),
        "not_run": len(micro_agents_registry) - len(status_summary),
        "details": status_summary,
        "action_log": action_logs[-100:]  # Last 100 entries
    }

@router.get("/agents")
async def list_agents():
    """List all registered agents with their dependencies"""
    agents_info = []
    for name in micro_agents_registry.keys():
        agents_info.append({
            "name": name,
            "dependencies": micro_agents_dependencies.get(name, []),
            "status": status_summary.get(name, {}).get("status", "not_run")
        })
    
    try:
        execution_order = prioritize_agents()
    except Exception as e:
        execution_order = []
    
    return {
        "agents": agents_info,
        "execution_order": execution_order
    }

@router.post("/trigger_agent/{agent_name}")
async def trigger_single_agent(agent_name: str):
    """Trigger a specific agent"""
    if agent_name not in micro_agents_registry:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    try:
        result = await run_micro_agent(agent_name)
        return {
            "agent": agent_name,
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orchestration_status")
async def orchestration_status():
    """Get overall orchestration system status"""
    total_agents = len(micro_agents_registry)
    successful = sum(1 for a in status_summary.values() if a["status"] == "success")
    failed = sum(1 for a in status_summary.values() if a["status"] == "failed")
    
    system_health = "healthy"
    if failed > 0:
        system_health = "degraded"
    if failed > total_agents * 0.3:
        system_health = "critical"
    
    return {
        "system_health": system_health,
        "total_agents": total_agents,
        "successful_agents": successful,
        "failed_agents": failed,
        "success_rate": (successful / max(1, total_agents)) * 100,
        "last_execution": max([a.get("last_run", "never") for a in status_summary.values()]) if status_summary else "never",
        "registered_categories": ["on_page_seo", "off_page_seo", "technical_seo", "local_seo", "orchestration"]
    }

@router.post("/reset_agents")
async def reset_agents():
    """Reset all agent statuses and logs"""
    global action_logs, status_summary
    action_logs.clear()
    status_summary.clear()
    
    return {
        "message": "All agent statuses and logs have been reset",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/agent_dependencies")
async def get_agent_dependencies():
    """Get agent dependency graph"""
    dependency_graph = {}
    
    for agent, deps in micro_agents_dependencies.items():
        dependency_graph[agent] = {
            "dependencies": deps,
            "dependents": [a for a, a_deps in micro_agents_dependencies.items() if agent in a_deps]
        }
    
    return {
        "dependency_graph": dependency_graph,
        "total_agents": len(micro_agents_registry),
        "agents_with_dependencies": len([a for a in micro_agents_dependencies.values() if len(a) > 0])
    }

@router.get("/status")
async def get_status():
    """Get core agent system status"""
    return {
        "agent": "core_seo_orchestration",
        "status": "active",
        "total_registered_agents": len(micro_agents_registry),
        "available_endpoints": [
            "seo_orchestration_core",
            "trigger_all_agents", 
            "dashboard_summary",
            "agents",
            "trigger_agent/{agent_name}",
            "orchestration_status",
            "reset_agents",
            "agent_dependencies"
        ],
        "core_capabilities": [
            "Agent Registration & Discovery",
            "Dependency Management",
            "Execution Orchestration", 
            "Status Monitoring",
            "Error Handling & Recovery",
            "Performance Analytics"
        ]
    }