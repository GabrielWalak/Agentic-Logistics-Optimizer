#!/usr/bin/env python3
"""
Pre-Deployment Checklist
Validates all components before AWS deployment
"""

import os
import sys
from pathlib import Path

def check_file_exists(path: str, name: str) -> bool:
    """Check if required file exists"""
    if Path(path).exists():
        print(f"✅ {name}: {path}")
        return True
    else:
        print(f"❌ {name}: NOT FOUND - {path}")
        return False

def check_env_var(var_name: str) -> bool:
    """Check if environment variable is set"""
    value = os.getenv(var_name)
    if value:
        masked = value[:20] + "..." if len(value) > 20 else value
        print(f"✅ {var_name}: {masked}")
        return True
    else:
        print(f"❌ {var_name}: NOT SET")
        return False

def main():
    print("=" * 70)
    print(" AGENTICAI - PRE-DEPLOYMENT CHECKLIST")
    print("=" * 70)
    print()
    
    checks_passed = 0
    checks_total = 0
    
    # === Check Files ===
    print("📋 Required Files:")
    print("-" * 70)
    required_files = [
        ("Dockerfile", "Docker image definition"),
        ("docker-compose.yml", "Docker Compose configuration"),
        ("requirements.txt", "Python dependencies"),
        (".dockerignore", "Docker build context exclusions"),
        (".env.example", "Environment template"),
        ("run.sh", "Container entry point"),
        ("app.py", "Main application"),
        ("pydantic_agents.py", "Multi-agent orchestration"),
        ("scenarios_examples.py", "Test scenarios"),
        ("prompt_engineering.py", "Prompt optimization & grading"),
        ("BUILD_INSTRUCTIONS.md", "Build documentation"),
        ("AWS_DEPLOYMENT.md", "AWS deployment guide"),
        ("QUICK_START_AWS.md", "Quick start guide"),
    ]
    
    for filename, description in required_files:
        checks_total += 1
        if check_file_exists(filename, description):
            checks_passed += 1
    
    print()
    
    # === Check Environment ===
    print("🔐 Environment Setup:")
    print("-" * 70)
    env_vars = ["GITHUB_TOKEN", "GITHUB_MODEL"]
    
    for var in env_vars:
        checks_total += 1
        if check_env_var(var):
            checks_passed += 1
    
    print()
    
    # === Check Docker ===
    print("🐳 Docker Setup:")
    print("-" * 70)
    checks_total += 1
    try:
        import docker
        print("✅ Docker Python SDK: installed")
        checks_passed += 1
    except ImportError:
        print("❌ Docker Python SDK: NOT INSTALLED (pip install docker)")
    
    # === Check Python Dependencies ===
    print()
    print("📦 Python Dependencies:")
    print("-" * 70)
    deps = ["openai", "pydantic", "python-dotenv", "redis", "langsmith"]
    
    for dep in deps:
        checks_total += 1
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep}: installed")
            checks_passed += 1
        except ImportError:
            print(f"❌ {dep}: NOT INSTALLED")
    
    print()
    
    # === Summary ===
    print("=" * 70)
    print(f" SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)
    print()
    
    if checks_passed == checks_total:
        print("🎉 All checks passed! Ready for deployment!")
        print()
        print("Next steps:")
        print("1. Read: QUICK_START_AWS.md")
        print("2. Create EC2 instance (t3.micro, Ubuntu 24.04)")
        print("3. SSH into instance")
        print("4. Run: curl ... | bash (or manual deploy)")
        print()
        return 0
    else:
        failed = checks_total - checks_passed
        print(f"⚠️  {failed} check(s) failed. Fix issues before deployment.")
        print()
        print("To fix:")
        if not os.getenv("GITHUB_TOKEN"):
            print("  - Set GITHUB_TOKEN: export GITHUB_TOKEN=your_token")
            print("    Or create .env file with: GITHUB_TOKEN=...")
        print("  - Ensure all required files are present")
        print("  - Run: pip install -r requirements.txt")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
