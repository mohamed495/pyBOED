"""
Benchmark: Posterior Inference with LIS
========================================

Comprehensive benchmark comparing posterior computation methods:
- Full posterior computation
- LIS-accelerated posterior

Problem configurations:
- Parameter dimensions: 50, 100, 200, 500, 1000
- Observation dimensions: 10, 20, 50
- LIS ranks: 5, 10, 20, 50

Metrics:
- Posterior computation time
- LIS construction time
- Mean and covariance errors
- Information gain preservation
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


def create_inverse_problem(n_params: int, n_obs: int, rank_H: int = None):
    """
    Create a synthetic linear Gaussian inverse problem.
    
    Parameters:
    -----------
    n_params : int
        Parameter dimension
    n_obs : int
        Observation dimension
    rank_H : int, optional
        Effective rank of forward operator (default: min(n_obs, n_params)//2)
    
    Returns:
    --------
    problem : dict
        Problem components (H, R, mu_prior, Sigma_prior, y_obs, theta_true)
    """
    if rank_H is None:
        rank_H = min(n_obs, n_params) // 2
    
    np.random.seed(42)
    
    # Forward operator (low-rank)
    U_H = np.random.randn(n_obs, rank_H)
    V_H = np.random.randn(n_params, rank_H)
    H = U_H @ V_H.T / np.sqrt(rank_H)
    
    # Prior
    mu_prior = np.zeros(n_params)
    
    # Smooth prior covariance
    x = np.linspace(0, 1, n_params)
    Sigma_prior = np.zeros((n_params, n_params))
    length_scale = 0.1
    for i in range(n_params):
        for j in range(n_params):
            Sigma_prior[i, j] = np.exp(-0.5 * ((x[i] - x[j]) / length_scale)**2)
    
    # Add small diagonal for numerical stability
    Sigma_prior += 1e-6 * np.eye(n_params)
    
    # Observation noise
    sigma_noise = 0.1
    R = sigma_noise**2 * np.eye(n_obs)
    
    # Generate truth and observations
    theta_true = np.random.multivariate_normal(mu_prior, Sigma_prior)
    y_obs = H @ theta_true + np.random.randn(n_obs) * sigma_noise
    
    return {
        'H': H,
        'R': R,
        'mu_prior': mu_prior,
        'Sigma_prior': Sigma_prior,
        'y_obs': y_obs,
        'theta_true': theta_true,
        'n_params': n_params,
        'n_obs': n_obs,
        'rank_H': rank_H
    }


def benchmark_full_posterior(problem: dict, n_trials: int = 3) -> Dict:
    """
    Benchmark full posterior computation.
    """
    from boed.inference import LinearGaussianModel
    import psutil
    import os
    
    print(f"\n  Full Posterior: n_params={problem['n_params']}, "
          f"n_obs={problem['n_obs']}")
    
    try:
        # Create model
        lgm = LinearGaussianModel(
            A=problem['H'],
            Sigma_noise=problem['R'],
            mu_prior=problem['mu_prior'],
            Sigma_prior=problem['Sigma_prior']
        )
        
        # Measure memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2
        
        # Time posterior computation
        times = []
        for _ in range(n_trials):
            start = time.time()
            mu_post, Sigma_post = lgm.posterior(problem['y_obs'])
            times.append(time.time() - start)
        
        time_mean = np.mean(times)
        time_std = np.std(times)
        
        mem_after = process.memory_info().rss / 1024**2
        mem_used = mem_after - mem_before
        
        # Information metrics
        from boed.design.criteria import DesignCriteria
        eig = DesignCriteria.EIG(Sigma_post, problem['Sigma_prior'])
        
        prior_trace = np.trace(problem['Sigma_prior'])
        post_trace = np.trace(Sigma_post)
        uncertainty_reduction = (prior_trace - post_trace) / prior_trace
        
        # Error vs truth
        error_truth = np.linalg.norm(mu_post - problem['theta_true'])
        
        print(f"    ✓ Time: {time_mean:.4f}±{time_std:.4f}s, "
              f"EIG: {eig:.3f}, Uncertainty: {uncertainty_reduction:.1%}")
        
        return {
            'method': 'full',
            'n_params': problem['n_params'],
            'n_obs': problem['n_obs'],
            'time_mean': time_mean,
            'time_std': time_std,
            'memory_mb': mem_used,
            'eig': eig,
            'uncertainty_reduction': uncertainty_reduction,
            'error_truth': error_truth,
            'mu_post': mu_post,
            'Sigma_post': Sigma_post,
            'success': True
        }
        
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        return {
            'method': 'full',
            'n_params': problem['n_params'],
            'success': False,
            'error': str(e)
        }


def benchmark_lis_posterior(
    problem: dict,
    lis_rank: int,
    n_samples: int = 500,
    n_trials: int = 3
) -> Dict:
    """
    Benchmark LIS-accelerated posterior computation.
    """
    from boed.integration import HybridBayesianInference
    from boed.design.criteria import DesignCriteria
    import psutil
    import os
    
    print(f"\n  LIS Posterior: n_params={problem['n_params']}, "
          f"n_obs={problem['n_obs']}, lis_rank={lis_rank}")
    
    try:
        # Create hybrid inference
        inference = HybridBayesianInference(
            forward_operator=problem['H'],
            noise_covariance=problem['R'],
            prior_mean=problem['mu_prior'],
            prior_covariance=problem['Sigma_prior'],
            lis_rank=lis_rank
        )
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2
        
        # Time LIS identification (one-time cost)
        start = time.time()
        inference.identify_informative_subspace(
            observations=problem['y_obs'],
            n_samples=n_samples,
            method='gradient',
            verbose=False
        )
        lis_build_time = time.time() - start
        
        # Time posterior computation (repeated)
        times = []
        for _ in range(n_trials):
            start = time.time()
            mu_post_lis, Sigma_post_lis = inference.posterior_in_subspace(
                problem['y_obs']
            )
            times.append(time.time() - start)
        
        time_mean = np.mean(times)
        time_std = np.std(times)
        
        # Total time (including LIS construction amortized over trials)
        total_time = lis_build_time + time_mean
        
        mem_after = process.memory_info().rss / 1024**2
        mem_used = mem_after - mem_before
        
        # Reconstruct for comparison
        mu_post_reconstructed = inference.reconstruct_from_subspace(mu_post_lis)
        
        # Compare with full posterior (if available in problem)
        if 'mu_post_full' in problem and 'Sigma_post_full' in problem:
            error_mean = np.linalg.norm(
                mu_post_reconstructed - problem['mu_post_full']
            ) / np.linalg.norm(problem['mu_post_full'])
        else:
            error_mean = None
        
        # Information in LIS subspace
        U_lis = inference._get_lis_basis()
        Sigma_prior_lis = U_lis.T @ problem['Sigma_prior'] @ U_lis
        eig_lis = DesignCriteria.EIG(Sigma_post_lis, Sigma_prior_lis)
        
        # Error vs truth
        error_truth = np.linalg.norm(mu_post_reconstructed - problem['theta_true'])
        
        # Variance captured
        variance_ratio = inference._compute_variance_ratio()
        
        print(f"    ✓ Build: {lis_build_time:.2f}s, Posterior: {time_mean:.4f}±{time_std:.4f}s, "
              f"Variance: {variance_ratio:.1%}")
        
        return {
            'method': 'lis',
            'n_params': problem['n_params'],
            'n_obs': problem['n_obs'],
            'lis_rank': lis_rank,
            'lis_build_time': lis_build_time,
            'time_mean': time_mean,
            'time_std': time_std,
            'total_time': total_time,
            'memory_mb': mem_used,
            'eig_lis': eig_lis,
            'variance_ratio': variance_ratio,
            'error_mean': error_mean,
            'error_truth': error_truth,
            'success': True
        }
        
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        return {
            'method': 'lis',
            'n_params': problem['n_params'],
            'success': False,
            'error': str(e)
        }


def run_benchmark_suite():
    """Run complete benchmark suite."""
    
    print("="*70)
    print("Posterior Inference with LIS Benchmark")
    print("="*70)
    
    results = []
    
    # Configuration
    param_dims = [50, 100, 200, 500]
    obs_dims = [10, 20, 50]
    lis_ranks = [5, 10, 20]
    
    for n_params in param_dims:
        for n_obs in obs_dims:
            if n_obs <= n_params // 2:  # Ensure underdetermined problem
                
                print(f"\n{'='*70}")
                print(f"Problem: n_params={n_params}, n_obs={n_obs}")
                print(f"{'='*70}")
                
                # Create problem
                problem = create_inverse_problem(n_params, n_obs)
                
                # Benchmark full posterior
                result_full = benchmark_full_posterior(problem)
                results.append(result_full)
                
                # Store full posterior for comparison
                if result_full['success']:
                    problem['mu_post_full'] = result_full['mu_post']
                    problem['Sigma_post_full'] = result_full['Sigma_post']
                
                # Benchmark LIS for different ranks
                for lis_rank in lis_ranks:
                    if lis_rank <= n_params // 2:  # Reasonable LIS rank
                        result_lis = benchmark_lis_posterior(
                            problem, lis_rank, n_samples=300
                        )
                        results.append(result_lis)
    
    return results


def save_and_visualize(results: List[Dict], output_dir: Path):
    """Save results and create plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove large arrays before saving JSON
    results_json = []
    for r in results:
        r_copy = r.copy()
        r_copy.pop('mu_post', None)
        r_copy.pop('Sigma_post', None)
        results_json.append(r_copy)
    
    json_path = output_dir / 'benchmark_posterior_inference.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✓ Results saved: {json_path}")
    
    # Create plots
    create_plots(results, output_dir)


def create_plots(results: List[Dict], output_dir: Path):
    """Create visualization plots."""
    
    # Filter successful results
    results_success = [r for r in results if r.get('success', False)]
    full_results = [r for r in results_success if r['method'] == 'full']
    lis_results = [r for r in results_success if r['method'] == 'lis']
    
    if not full_results or not lis_results:
        print("⚠ Insufficient data for plots")
        return
    
    # Plot 1: Time vs Problem Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Full posterior time
    n_params_full = [r['n_params'] for r in full_results]
    time_full = [r['time_mean'] for r in full_results]
    ax1.plot(n_params_full, time_full, 'o-', linewidth=2, markersize=8,
            label='Full posterior', color='blue')
    
    # LIS posterior time (by rank)
    for rank in sorted(set(r['lis_rank'] for r in lis_results)):
        lis_rank = [r for r in lis_results if r['lis_rank'] == rank]
        n_params_lis = [r['n_params'] for r in lis_rank]
        time_lis = [r['time_mean'] for r in lis_rank]
        ax1.plot(n_params_lis, time_lis, 's--', linewidth=2, markersize=6,
                label=f'LIS (rank={rank})', alpha=0.7)
    
    ax1.set_xlabel('Parameter Dimension', fontsize=12)
    ax1.set_ylabel('Posterior Time (s)', fontsize=12)
    ax1.set_title('Posterior Computation Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Speedup
    # Group by n_params and compute speedup
    for n_params in sorted(set(r['n_params'] for r in full_results)):
        full = [r for r in full_results if r['n_params'] == n_params]
        if full:
            time_f = full[0]['time_mean']
            
            lis = [r for r in lis_results if r['n_params'] == n_params]
            if lis:
                ranks = [r['lis_rank'] for r in lis]
                speedups = [time_f / r['time_mean'] for r in lis]
                
                ax2.plot(ranks, speedups, 'o-', linewidth=2, markersize=6,
                        label=f'n={n_params}', alpha=0.7)
    
    ax2.set_xlabel('LIS Rank', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('LIS Speedup vs Rank', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lis_time_speedup.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_dir / 'lis_time_speedup.png'}")
    
    # Plot 2: Accuracy vs Speedup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter LIS results with error measurements
    lis_with_error = [r for r in lis_results if r['error_mean'] is not None]
    
    if lis_with_error:
        # Compute speedups
        speedups = []
        errors = []
        sizes = []
        
        for r in lis_with_error:
            # Find corresponding full result
            full = [f for f in full_results 
                   if f['n_params'] == r['n_params'] and f['n_obs'] == r['n_obs']]
            if full:
                speedup = full[0]['time_mean'] / r['time_mean']
                speedups.append(speedup)
                errors.append(r['error_mean'] * 100)
                sizes.append(r['n_params'])
        
        scatter = ax1.scatter(speedups, errors, c=sizes, s=100, alpha=0.7,
                            cmap='viridis')
        plt.colorbar(scatter, ax=ax1, label='Problem Size')
        ax1.set_xlabel('Speedup Factor', fontsize=12)
        ax1.set_ylabel('Relative Error (%)', fontsize=12)
        ax1.set_title('Accuracy vs Speed Tradeoff', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
    
    # Variance captured vs LIS rank
    ranks = [r['lis_rank'] for r in lis_results]
    variance = [r['variance_ratio'] * 100 for r in lis_results]
    n_params_list = [r['n_params'] for r in lis_results]
    
    scatter2 = ax2.scatter(ranks, variance, c=n_params_list, s=100, alpha=0.7,
                          cmap='plasma')
    plt.colorbar(scatter2, ax=ax2, label='Problem Size')
    ax2.axhline(99, color='r', linestyle='--', linewidth=2, alpha=0.5,
               label='99% threshold')
    ax2.set_xlabel('LIS Rank', fontsize=12)
    ax2.set_ylabel('Variance Captured (%)', fontsize=12)
    ax2.set_title('Information Preservation', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 101])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lis_accuracy_variance.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_dir / 'lis_accuracy_variance.png'}")
    
    # Plot 3: Memory Usage
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_params_full = [r['n_params'] for r in full_results]
    mem_full = [r['memory_mb'] for r in full_results]
    ax.plot(n_params_full, mem_full, 'o-', linewidth=2, markersize=8,
           label='Full posterior', color='blue')
    
    for rank in sorted(set(r['lis_rank'] for r in lis_results)):
        lis_rank = [r for r in lis_results if r['lis_rank'] == rank]
        n_params_lis = [r['n_params'] for r in lis_rank]
        mem_lis = [r['memory_mb'] for r in lis_rank]
        ax.plot(n_params_lis, mem_lis, 's--', linewidth=2, markersize=6,
               label=f'LIS (rank={rank})', alpha=0.7)
    
    ax.set_xlabel('Parameter Dimension', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lis_memory.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_dir / 'lis_memory.png'}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    
    results_success = [r for r in results if r.get('success', False)]
    full_results = [r for r in results_success if r['method'] == 'full']
    lis_results = [r for r in results_success if r['method'] == 'lis']
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {len(results_success)}")
    print(f"Failed: {len(results) - len(results_success)}")
    
    print(f"\nFull Posterior:")
    print(f"  Tests: {len(full_results)}")
    if full_results:
        times = [r['time_mean'] for r in full_results]
        print(f"  Time range: {min(times):.4f}s - {max(times):.4f}s")
        
        eigs = [r['eig'] for r in full_results]
        print(f"  EIG range: {min(eigs):.2f} - {max(eigs):.2f} nats")
    
    print(f"\nLIS Posterior:")
    print(f"  Tests: {len(lis_results)}")
    if lis_results:
        times = [r['time_mean'] for r in lis_results]
        print(f"  Time range: {min(times):.4f}s - {max(times):.4f}s")
        
        # Compute speedups
        speedups = []
        for r_lis in lis_results:
            full = [f for f in full_results 
                   if f['n_params'] == r_lis['n_params'] and 
                      f['n_obs'] == r_lis['n_obs']]
            if full:
                speedup = full[0]['time_mean'] / r_lis['time_mean']
                speedups.append(speedup)
        
        if speedups:
            print(f"  Speedup range: {min(speedups):.1f}x - {max(speedups):.1f}x")
            print(f"  Average speedup: {np.mean(speedups):.1f}x")
        
        variance = [r['variance_ratio'] for r in lis_results]
        print(f"  Variance captured: {min(variance):.1%} - {max(variance):.1%}")
    
    print("="*70)


def main():
    """Run the benchmark."""
    
    results = run_benchmark_suite()
    
    output_dir = Path('results/benchmarks')
    save_and_visualize(results, output_dir)
    
    print_summary(results)


if __name__ == "__main__":
    main()
