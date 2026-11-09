#!/usr/bin/env python3
"""
Orchestrator: Run both SegRDNet and SoftCAM on APTOS 2019
Run in Kaggle notebook to compare both architectures.

Usage in Kaggle:
    # Clone both repos
    !git clone https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization.git segrdnet_repo
    !git clone https://github.com/chihchiehchen/SoftCAM.git softcam_repo
    
    # Run comparison
    %cd /kaggle/working
    !python run_both_aptos.py --epochs 15 --batch-size 16
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime


def run_command(cmd, cwd=None, env=None):
    """Run a command and stream output"""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    process = subprocess.Popen(
        cmd, 
        cwd=cwd, 
        env=env,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    output_lines = []
    for line in process.stdout:
        print(line.rstrip())
        output_lines.append(line)
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\n❌ Command failed with return code {process.returncode}")
        return False, output_lines
    
    return True, output_lines


def clone_repo(repo_url, dir_name):
    """Clone a git repository if not exists"""
    if not os.path.exists(dir_name):
        print(f"Cloning {repo_url}...")
        success, _ = run_command(['git', 'clone', repo_url, dir_name])
        return success
    else:
        print(f"Directory {dir_name} already exists, skipping clone")
        return True


def main():
    parser = argparse.ArgumentParser(description='Run both SegRDNet and SoftCAM on APTOS')
    parser.add_argument('--epochs', default=15, type=int, help='Epochs per model (10-20)')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--img-size', default=512, type=int)
    parser.add_argument('--data-dir', default='/kaggle/input/competitions/aptos2019-blindness-detection', type=str)
    parser.add_argument('--skip-segrdnet', action='store_true', help='Skip SegRDNet')
    parser.add_argument('--skip-softcam', action='store_true', help='Skip SoftCAM')
    parser.add_argument('--softcam-network', default='resnet50', type=str)
    parser.add_argument('--output-base', default='/kaggle/working/aptos_comparison', type=str)
    
    args = parser.parse_args()
    
    base_dir = '/kaggle/working'
    os.makedirs(args.output_base, exist_ok=True)
    
    results = {}
    
    # ============ SegRDNet ============
    if not args.skip_segrdnet:
        print("\n" + "="*80)
        print("RUNNING SEGRDNET")
        print("="*80)
        
        segrdnet_dir = os.path.join(base_dir, 'segrdnet_repo')
        success = clone_repo(
            'https://github.com/chihchiehchen/Beyond-Classification-Elaborating-Network-Predictions-for-Better-Weakly-Supervised-Quantization.git',
            segrdnet_dir
        )
        
        if success:
            script_path = os.path.join(base_dir, 'run_segrdnet_aptos.py')
            if os.path.exists(script_path):
                cmd = [
                    sys.executable, script_path,
                    '--data-dir', args.data_dir,
                    '--epochs', str(args.epochs),
                    '--batch-size', str(args.batch_size),
                    '--img-size', str(args.img_size),
                    '--output', os.path.join(args.output_base, 'segrdnet')
                ]
                success, output = run_command(cmd, cwd=segrdnet_dir)
                
                # Parse results from output
                results['segrdnet'] = {'success': success}
                for line in output:
                    if 'Best QWK:' in line:
                        results['segrdnet']['best_qwk'] = float(line.split('Best QWK:')[1].split()[0])
                    if 'Best AUC:' in line:
                        results['segrdnet']['best_auc'] = float(line.split('Best AUC:')[1].split()[0])
            else:
                print(f"Script not found: {script_path}")
                results['segrdnet'] = {'success': False, 'error': 'Script not found'}
        else:
            results['segrdnet'] = {'success': False, 'error': 'Clone failed'}
    
    # ============ SoftCAM ============
    if not args.skip_softcam:
        print("\n" + "="*80)
        print("RUNNING SOFTCAM")
        print("="*80)
        
        softcam_dir = os.path.join(base_dir, 'softcam_repo')
        success = clone_repo(
            'https://github.com/chihchiehchen/SoftCAM.git',
            softcam_dir
        )
        
        if success:
            script_path = os.path.join(base_dir, 'run_softcam_aptos.py')
            if os.path.exists(script_path):
                cmd = [
                    sys.executable, script_path,
                    '--data-dir', args.data_dir,
                    '--network', args.softcam_network,
                    '--epochs', str(args.epochs),
                    '--batch-size', str(args.batch_size),
                    '--img-size', str(args.img_size),
                    '--output', os.path.join(args.output_base, 'softcam')
                ]
                success, output = run_command(cmd, cwd=softcam_dir)
                
                # Parse results
                results['softcam'] = {'success': success, 'network': args.softcam_network}
                for line in output:
                    if 'Best validation QWK:' in line:
                        results['softcam']['best_qwk'] = float(line.split('Best validation QWK:')[1].split()[0])
            else:
                print(f"Script not found: {script_path}")
                results['softcam'] = {'success': False, 'error': 'Script not found'}
        else:
            results['softcam'] = {'success': False, 'error': 'Clone failed'}
    
    # ============ Summary ============
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Success':<10} {'Best QWK':<12} {'Best AUC':<12}")
    print("-"*60)
    
    for model_name, result in results.items():
        if model_name == 'segrdnet':
            qwk = result.get('best_qwk', 'N/A')
            auc = result.get('best_auc', 'N/A')
            print(f"{'SegRDNet':<20} {str(result.get('success', False)):<10} {qwk:<12} {auc:<12}")
        elif model_name == 'softcam':
            qwk = result.get('best_qwk', 'N/A')
            network = result.get('network', 'N/A')
            print(f"{'SoftCAM ('+network+')':<20} {str(result.get('success', False)):<10} {qwk:<12} {'N/A':<12}")
    
    # Save results
    results_path = os.path.join(args.output_base, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Check submissions
    print("\nSubmission files:")
    for model_name in results:
        sub_path = os.path.join(args.output_base, model_name, 'submission.csv')
        if os.path.exists(sub_path):
            print(f"  ✓ {model_name}: {sub_path}")
        else:
            # Find submission in subdirectories
            for root, dirs, files in os.walk(os.path.join(args.output_base, model_name)):
                if 'submission.csv' in files:
                    print(f"  ✓ {model_name}: {os.path.join(root, 'submission.csv')}")
                    break
            else:
                print(f"  ✗ {model_name}: submission.csv not found")


if __name__ == '__main__':
    main()