"""
Pre-Deployment Checklist Script
Run this before deploying to Render to ensure everything is ready
"""

import os
import sys

def check_file_exists(filepath, critical=True):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    level = "CRITICAL" if critical and not exists else "OK" if exists else "WARNING"
    print(f"  [{status}] {filepath} - {level}")
    return exists

def check_model_size(filepath):
    """Check model file size"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"      Size: {size:.2f} MB")
        if size > 100:
            print(f"      ⚠ WARNING: Large file! Consider Git LFS")
        return True
    return False

def main():
    print("="*60)
    print("RENDER DEPLOYMENT CHECKLIST")
    print("="*60)
    
    all_checks_passed = True
    
    # Check critical files
    print("\n[1] Checking Critical Files...")
    critical_files = [
        "app.py",
        "requirements.txt",
        "render.yaml",
        "README.md"
    ]
    
    for file in critical_files:
        if not check_file_exists(file, critical=True):
            all_checks_passed = False
    
    # Check model files
    print("\n[2] Checking Model Files...")
    model_files = [
        "app/model/model.pkl",
        "app/model/scaler.pkl"
    ]
    
    for file in model_files:
        if check_file_exists(file, critical=True):
            check_model_size(file)
        else:
            all_checks_passed = False
    
    # Check app structure
    print("\n[3] Checking App Structure...")
    required_dirs = [
        "app",
        "app/templates",
        "app/static",
        "app/model"
    ]
    
    for directory in required_dirs:
        if not os.path.isdir(directory):
            print(f"  [✗] Directory missing: {directory} - CRITICAL")
            all_checks_passed = False
        else:
            print(f"  [✓] {directory} - OK")
    
    # Check requirements.txt content
    print("\n[4] Checking requirements.txt...")
    required_packages = ["Flask", "gunicorn", "pandas", "scikit-learn", "joblib"]
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
            for package in required_packages:
                if package.lower() in content.lower():
                    print(f"  [✓] {package} found")
                else:
                    print(f"  [✗] {package} MISSING - CRITICAL")
                    all_checks_passed = False
    except FileNotFoundError:
        print("  [✗] requirements.txt not found - CRITICAL")
        all_checks_passed = False
    
    # Check app.py for PORT handling
    print("\n[5] Checking app.py Configuration...")
    try:
        with open("app.py", "r") as f:
            content = f.read()
            if "os.environ.get('PORT'" in content or "os.getenv('PORT'" in content:
                print("  [✓] PORT environment variable handling found")
            else:
                print("  [⚠] PORT handling not found - May cause issues on Render")
                print("      Add: port = int(os.environ.get('PORT', 5000))")
    except FileNotFoundError:
        print("  [✗] app.py not found - CRITICAL")
        all_checks_passed = False
    
    # Check .gitignore
    print("\n[6] Checking .gitignore...")
    if check_file_exists(".gitignore", critical=False):
        with open(".gitignore", "r") as f:
            content = f.read()
            if "!app/model/*.pkl" in content or "!*.pkl" in content:
                print("      [✓] Model files are NOT ignored (correct)")
            else:
                print("      [⚠] Warning: Model files might be ignored by git")
                print("          Add '!app/model/*.pkl' to .gitignore")
    
    # Git status check
    print("\n[7] Checking Git Status...")
    git_status = os.system("git status > nul 2>&1")
    if git_status == 0:
        print("  [✓] Git repository initialized")
        
        # Check if model files are tracked
        print("\n  Checking if model files are committed...")
        for model_file in model_files:
            result = os.system(f"git ls-files {model_file} > nul 2>&1")
            if result == 0:
                print(f"    [✓] {model_file} is tracked by git")
            else:
                print(f"    [✗] {model_file} NOT tracked - CRITICAL")
                print(f"        Run: git add {model_file}")
                all_checks_passed = False
    else:
        print("  [✗] Not a git repository - CRITICAL")
        print("      Run: git init")
        all_checks_passed = False
    
    # Final summary
    print("\n" + "="*60)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED - READY FOR DEPLOYMENT!")
        print("="*60)
        print("\nNext steps:")
        print("1. Commit all changes: git add . && git commit -m 'Prepare for deployment'")
        print("2. Push to GitHub: git push origin main")
        print("3. Go to https://render.com and create a new Web Service")
        print("4. Connect your GitHub repository")
        print("5. Set Root Directory to: fraud_detection_web")
        print("6. Deploy!")
    else:
        print("✗ SOME CHECKS FAILED - FIX ISSUES BEFORE DEPLOYING")
        print("="*60)
        print("\nPlease fix the issues marked as CRITICAL above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
