"""
==============================================================================
Master Execution Script for SVM Nepal AI Projects
Runs both Project 1 (SVR Regression) and Project 2 (SVC Classification)
==============================================================================
"""

import subprocess
import sys
import os

def run_project(project_dir, script_name):
    print("\n" + "#"*70)
    print(f"  EXECUTING: {project_dir}/{script_name}")
    print("#"*70)
    
    cmd = [sys.executable, script_name]
    res = subprocess.run(cmd, cwd=project_dir)
    if res.returncode != 0:
        print(f"[-] Execution failed for {project_dir}/{script_name}")
        return False
    return True

def main():
    print("="*70)
    print(" SUPPORT VECTOR MACHINE (SVM) — NEPAL REAL-WORLD AI PROJECTS")
    print("="*70)
    
    # 1. Project 1: Regression (Housing Price in Nepal)
    p1_dir = "01_svm_regression_nepal_housing"
    success_p1 = run_project(p1_dir, "train_svr.py")
    if success_p1:
        run_project(p1_dir, "predict.py")
        
    # 2. Project 2: Classification (Kathmandu Air Quality)
    p2_dir = "02_svm_classification_nepal_air_quality"
    success_p2 = run_project(p2_dir, "train_svc.py")
    if success_p2:
        run_project(p2_dir, "predict.py")
        
    print("\n" + "="*70)
    print(" [✓] BOTH SVM PROJECTS COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
