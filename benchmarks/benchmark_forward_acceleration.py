"""
Benchmark: Forward Model Acceleration with POD
===============================================

Comprehensive benchmark measuring POD speedup for different configurations:
- Spatial resolutions: N = 50, 100, 200, 500, 1000, 2000
- Time steps: 10, 50, 100, 200
- POD energy thresholds: 0.99, 0.999, 0.9999

Metrics:
- Offline time (POD construction)
- Online time per evaluation
- Speedup factor
- Accuracy (L2 error, max error)
- Memory usage
"""

import numpy as np
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure local pyBOED package is imported when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _json_default(obj):
    """JSON serializer for NumPy scalars/arrays."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def generate_training_snapshots(N: int, n_samples: int, n_steps: int):
    """
    Generate training snapshots for POD.
    
    Parameters:
    -----------
    N : int
        Spatial resolution
    n_samples : int
        Number of parameter samples
    n_steps : int
        Time steps per sample
    
    Returns:
    --------
    snapshots : list
        List of initial conditions for POD training
    """
    from boed.core import make_u0
    
    x = np.linspace(0, 1, N)
    snapshots = []
    
    # Generate varied initial conditions
    for i in range(n_samples):
        u0_type = ['gaussian', 'double_gaussian', 'sine'][i % 3]
        
        if u0_type == 'gaussian':
            mu = 0.3 + 0.4 * (i / n_samples)
            sigma = 0.05 + 0.1 * np.random.rand()
            u0 = make_u0(x, 'gaussian', center=mu, width=sigma)
        elif u0_type == 'double_gaussian':
            u0 = make_u0(x, 'double_gaussian')
        else:
            freq = 1 + 2 * np.random.rand()
            k = max(1, int(round(freq)))
            u0 = make_u0(x, 'sine', k=k)
        
        snapshots.append(u0)
    
    return snapshots


def benchmark_pod_construction(
    N: int,
    n_steps: int,
    energy_threshold: float,
    n_train_samples: int = 20
) -> Dict:
    """
    Benchmark POD model construction (offline phase).
    """
    from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
    from boed.integration import PODForwardModel
    import psutil
    import os
    
    print(f"\n  POD Construction: N={N}, steps={n_steps}, "
          f"threshold={energy_threshold}")
    
    try:
        # Create full model
        dt = 0.001 if N <= 500 else 0.002
        model = AdvectionDiffusion1D_CN(N=N, dt=dt, diffusivity=0.01, velocity=0.5)
        
        # Generate training data
        snapshots = generate_training_snapshots(N, n_train_samples, n_steps)
        
        # Measure memory and time for POD construction
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2
        
        pod_model = PODForwardModel(model, energy_threshold=energy_threshold)
        
        start_time = time.time()
        pod_model.build_reduced_model(
            parameter_samples=snapshots,
            n_steps=n_steps,
            verbose=False
        )
        construction_time = time.time() - start_time
        
        mem_after = process.memory_info().rss / 1024**2
        mem_used = mem_after - mem_before
        
        # Get POD properties
        n_modes = pod_model.n_modes
        cumulative_energy = np.cumsum(pod_model.pod.energy_ratio_)
        energy_captured = cumulative_energy[n_modes - 1]
        reduction_ratio = N / n_modes
        
        print(f"    ✓ Time: {construction_time:.2f}s, Modes: {n_modes}, "
              f"Energy: {energy_captured:.4f}, Memory: {mem_used:.1f}MB")
        
        return {
            'N': N,
            'n_steps': n_steps,
            'energy_threshold': energy_threshold,
            'construction_time': construction_time,
            'memory_mb': mem_used,
            'n_modes': n_modes,
            'reduction_ratio': reduction_ratio,
            'energy_captured': energy_captured,
            'success': True
        }
        
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        return {
            'N': N,
            'success': False,
            'error': str(e)
        }


def benchmark_pod_online(
    N: int,
    n_steps: int,
    energy_threshold: float,
    n_eval: int = 10
) -> Dict:
    """
    Benchmark POD online evaluations (speedup measurement).
    """
    from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
    from boed.integration import PODForwardModel
    from boed.core import make_u0
    
    print(f"\n  POD Online: N={N}, steps={n_steps}, threshold={energy_threshold}")
    
    try:
        # Setup
        dt = 0.001 if N <= 500 else 0.002
        model = AdvectionDiffusion1D_CN(N=N, dt=dt, diffusivity=0.01, velocity=0.5)
        
        # Build POD model
        snapshots = generate_training_snapshots(N, 20, n_steps)
        pod_model = PODForwardModel(model, energy_threshold=energy_threshold)
        pod_model.build_reduced_model(snapshots, n_steps, verbose=False)
        
        # Test initial condition
        x = np.linspace(0, 1, N)
        u0_test = make_u0(x, 'gaussian', center=0.5, width=0.08)
        
        # Time full model
        times_full = []
        for _ in range(n_eval):
            start = time.time()
            traj_full = model.evolve(u0_test, n_steps=n_steps)
            times_full.append(time.time() - start)
        
        time_full_mean = np.mean(times_full)
        time_full_std = np.std(times_full)
        
        # Time POD model
        times_pod = []
        for _ in range(n_eval):
            start = time.time()
            traj_pod = pod_model.evolve_reduced(u0_test, n_steps=n_steps, 
                                                method='galerkin')
            times_pod.append(time.time() - start)
        
        time_pod_mean = np.mean(times_pod)
        time_pod_std = np.std(times_pod)
        
        # Compute accuracy
        traj_full = model.evolve(u0_test, n_steps=n_steps)
        traj_pod = pod_model.evolve_reduced(u0_test, n_steps=n_steps, 
                                           method='galerkin')
        
        error = traj_full - traj_pod
        rel_error = np.linalg.norm(error) / np.linalg.norm(traj_full)
        max_error = np.max(np.abs(error))
        
        # Speedup
        speedup = time_full_mean / time_pod_mean
        
        print(f"    ✓ Full: {time_full_mean:.4f}s, POD: {time_pod_mean:.4f}s, "
              f"Speedup: {speedup:.1f}x, Error: {rel_error:.2e}")
        
        return {
            'N': N,
            'n_steps': n_steps,
            'energy_threshold': energy_threshold,
            'n_modes': pod_model.n_modes,
            'time_full_mean': time_full_mean,
            'time_full_std': time_full_std,
            'time_pod_mean': time_pod_mean,
            'time_pod_std': time_pod_std,
            'speedup': speedup,
            'rel_error': rel_error,
            'max_error': max_error,
            'success': True
        }
        
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        return {
            'N': N,
            'success': False,
            'error': str(e)
        }


def run_benchmark_suite():
    """Run complete benchmark suite."""
    
    print("="*70)
    print("POD Forward Model Acceleration Benchmark")
    print("="*70)
    
    results_construction = []
    results_online = []
    
    # Configuration
    sizes = [50, 100, 200, 500, 1000]
    n_steps_list = [10, 50, 100]
    thresholds = [0.99, 0.999, 0.9999]
    
    # Benchmark 1: POD Construction
    print("\n" + "="*70)
    print("BENCHMARK 1: POD Construction (Offline Phase)")
    print("="*70)
    
    for N in sizes:
        for n_steps in n_steps_list:
            for threshold in thresholds:
                result = benchmark_pod_construction(N, n_steps, threshold)
                results_construction.append(result)
    
    # Benchmark 2: POD Online Evaluation
    print("\n" + "="*70)
    print("BENCHMARK 2: POD Online Evaluation (Speedup)")
    print("="*70)
    
    # Use subset for online tests (more expensive)
    sizes_online = [100, 200, 500, 1000]
    n_steps_online = [50, 100]
    thresholds_online = [0.999, 0.9999]
    
    for N in sizes_online:
        for n_steps in n_steps_online:
            for threshold in thresholds_online:
                result = benchmark_pod_online(N, n_steps, threshold)
                results_online.append(result)
    
    return results_construction, results_online


def save_and_visualize(results_construction, results_online, output_dir: Path):
    """Save results and create plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    results = {
        'construction': results_construction,
        'online': results_online
    }
    
    json_path = output_dir / 'benchmark_forward_acceleration.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\n✓ Results saved: {json_path}")
    
    # Create plots
    create_plots(results_construction, results_online, output_dir)


def create_plots(results_construction, results_online, output_dir: Path):
    """Create visualization plots."""
    
    # Filter successful results
    constr_success = [r for r in results_construction if r.get('success', False)]
    online_success = [r for r in results_online if r.get('success', False)]
    
    if not constr_success or not online_success:
        print("⚠ Insufficient data for plots")
        return
    
    # Plot 1: Construction Time vs Problem Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for threshold in sorted(set(r['energy_threshold'] for r in constr_success)):
        data = [r for r in constr_success if r['energy_threshold'] == threshold]
        N_vals = [r['N'] for r in data]
        times = [r['construction_time'] for r in data]
        
        ax1.plot(N_vals, times, 'o-', linewidth=2, markersize=6,
                label=f'Threshold = {threshold}', alpha=0.7)
    
    ax1.set_xlabel('Spatial Resolution (N)', fontsize=12)
    ax1.set_ylabel('Construction Time (s)', fontsize=12)
    ax1.set_title('POD Construction Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Number of Modes vs Problem Size
    for threshold in sorted(set(r['energy_threshold'] for r in constr_success)):
        data = [r for r in constr_success if r['energy_threshold'] == threshold]
        N_vals = [r['N'] for r in data]
        modes = [r['n_modes'] for r in data]
        
        ax2.plot(N_vals, modes, 's-', linewidth=2, markersize=6,
                label=f'Threshold = {threshold}', alpha=0.7)
    
    ax2.set_xlabel('Spatial Resolution (N)', fontsize=12)
    ax2.set_ylabel('Number of POD Modes', fontsize=12)
    ax2.set_title('POD Compression', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pod_construction.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_dir / 'pod_construction.png'}")
    
    # Plot 3: Speedup vs Problem Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for threshold in sorted(set(r['energy_threshold'] for r in online_success)):
        data = [r for r in online_success if r['energy_threshold'] == threshold]
        N_vals = [r['N'] for r in data]
        speedups = [r['speedup'] for r in data]
        
        ax1.plot(N_vals, speedups, 'o-', linewidth=2, markersize=8,
                label=f'Threshold = {threshold}', alpha=0.7)
    
    # Theoretical speedup (linear model)
    N_theory = np.array(sorted(set(r['N'] for r in online_success)))
    speedup_theory = N_theory / 50  # Rough estimate
    ax1.plot(N_theory, speedup_theory, 'k--', linewidth=2, alpha=0.3,
            label='Theoretical (∝ N)')
    
    ax1.set_xlabel('Spatial Resolution (N)', fontsize=12)
    ax1.set_ylabel('Speedup Factor', fontsize=12)
    ax1.set_title('POD Speedup', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 4: Accuracy vs Speedup
    speedups = [r['speedup'] for r in online_success]
    errors = [r['rel_error'] * 100 for r in online_success]
    sizes = [r['N'] for r in online_success]
    
    scatter = ax2.scatter(speedups, errors, c=sizes, s=100, alpha=0.7,
                         cmap='viridis')
    plt.colorbar(scatter, ax=ax2, label='Problem Size (N)')
    
    ax2.set_xlabel('Speedup Factor', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title('Accuracy vs Speed Tradeoff', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pod_speedup_accuracy.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_dir / 'pod_speedup_accuracy.png'}")
    
    # Plot 5: Time Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by problem size
    for N in sorted(set(r['N'] for r in online_success)):
        data = [r for r in online_success if r['N'] == N]
        
        time_full = [r['time_full_mean'] for r in data]
        time_pod = [r['time_pod_mean'] for r in data]
        x_pos = np.arange(len(data))
        
        width = 0.35
        ax.bar(x_pos - width/2, time_full, width, label=f'Full (N={N})', 
               alpha=0.7)
        ax.bar(x_pos + width/2, time_pod, width, label=f'POD (N={N})',
               alpha=0.7)
    
    ax.set_ylabel('Time per Evaluation (s)', fontsize=12)
    ax.set_title('Full vs POD Evaluation Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pod_time_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_dir / 'pod_time_comparison.png'}")


def print_summary(results_construction, results_online):
    """Print summary statistics."""
    
    constr_success = [r for r in results_construction if r.get('success', False)]
    online_success = [r for r in results_online if r.get('success', False)]
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print("\nConstruction Phase:")
    print(f"  Tests run: {len(results_construction)}")
    print(f"  Successful: {len(constr_success)}")
    
    if constr_success:
        times = [r['construction_time'] for r in constr_success]
        print(f"  Time range: {min(times):.2f}s - {max(times):.2f}s")
        
        modes = [r['n_modes'] for r in constr_success]
        print(f"  Modes range: {min(modes)} - {max(modes)}")
        
        reductions = [r['reduction_ratio'] for r in constr_success]
        print(f"  Reduction ratio range: {min(reductions):.1f}x - {max(reductions):.1f}x")
    
    print("\nOnline Phase:")
    print(f"  Tests run: {len(results_online)}")
    print(f"  Successful: {len(online_success)}")
    
    if online_success:
        speedups = [r['speedup'] for r in online_success]
        print(f"  Speedup range: {min(speedups):.1f}x - {max(speedups):.1f}x")
        print(f"  Average speedup: {np.mean(speedups):.1f}x")
        
        errors = [r['rel_error'] for r in online_success]
        print(f"  Error range: {min(errors):.2e} - {max(errors):.2e}")
        print(f"  Average error: {np.mean(errors):.2e}")
    
    print("="*70)


def main():
    """Run the benchmark."""
    
    results_construction, results_online = run_benchmark_suite()
    
    output_dir = Path('results/benchmarks')
    save_and_visualize(results_construction, results_online, output_dir)
    
    print_summary(results_construction, results_online)


if __name__ == "__main__":
    main()
