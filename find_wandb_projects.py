#!/usr/bin/env python3
"""
Find available wandb projects and runs
"""

def find_wandb_projects():
    """Find all available wandb projects"""
    
    try:
        import wandb
    except ImportError:
        print("❌ wandb not available")
        return
    
    try:
        # Initialize wandb API
        api = wandb.Api()
        
        # Get current user info
        print("👤 Current user:", api.viewer.username)
        print("🏢 Current entity:", api.viewer.entity)
        print()
        
        # List all projects for current user
        print("📋 Available projects:")
        projects = api.projects()
        
        if not projects:
            print("❌ No projects found")
            return
        
        for i, project in enumerate(projects):
            print(f"{i+1}. {project.name}")
            
            # Get recent runs for this project
            runs = list(project.runs())[:3]  # Get first 3 runs
            if runs:
                print(f"   Recent runs:")
                for run in runs:
                    print(f"   - {run.id} ({run.name}) - {run.state}")
            print()
        
        # If we found projects, analyze the most recent one
        if projects:
            most_recent_project = projects[0]
            print(f"🔍 Let's analyze the most recent project: {most_recent_project.name}")
            
            runs = list(most_recent_project.runs())
            if runs:
                recent_run = runs[0]
                print(f"📊 Most recent run: {recent_run.id}")
                print(f"📅 Created: {recent_run.created_at}")
                print(f"📈 State: {recent_run.state}")
                
                # Show some config info
                if hasattr(recent_run, 'config'):
                    print("🎛️ Config preview:")
                    config_items = list(recent_run.config.items())[:5]
                    for key, value in config_items:
                        print(f"   {key}: {value}")
                
                print(f"\n💡 To analyze this run, use project name: '{most_recent_project.name}'")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you're logged into wandb and have projects")

if __name__ == "__main__":
    find_wandb_projects() 