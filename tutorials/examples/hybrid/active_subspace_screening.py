"""
Hybrid Example: Active Subspace Parameter Screening for BOED
=============================================================

This example demonstrates using Active Subspaces to identify important
parameters BEFORE running BOED:

1. Sample parameter space and evaluate a quantity of interest (QoI)
2. Compute gradients of QoI with respect to parameters
3. Build Active Subspace to identify important directions
4. Focus BOED on the active subspace (reduced dimensions)
5. Compare full-space vs subspace designs

Expected benefit: 
- Identify which parameters actually matter
- Focus sensors on informative parameters
- Reduce problem complexity before BOED
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure local pyBOED package is imported when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("="*70)
    print("Hybrid BOED Example: Active Subspace Screening")
    print("="*70)
    
    # ========================================================================
    # Step 1: Problem Setup
    # ========================================================================
    print("\n" + "="*70)
    print("Step 1: Multi-Parameter Problem Setup")
    print("="*70)
    
    from boed.pde.advection_diffusion import AdvectionDiffusion1D_CN
    from boed.core import make_u0
    
    # Create a problem with multiple parameters
    # We'll vary: initial condition parameters, diffusivity, velocity
    
    N = 100  # Spatial points
    dt = 0.005
    n_steps = 20
    
    print(f"PDE grid: N = {N}")
    print(f"Time steps: {n_steps}")
    
    # We'll work with 6 parameters:
    # [mu1, sigma1, mu2, sigma2, diffusivity, velocity]
    # for a double Gaussian initial condition
    
    n_params = 6
    print(f"\nParameter space dimension: {n_params}")
    print("Parameters:")
    print("  [0] mu1: first Gaussian center")
    print("  [1] sigma1: first Gaussian width")
    print("  [2] mu2: second Gaussian center")
    print("  [3] sigma2: second Gaussian width")
    print("  [4] diffusivity")
    print("  [5] velocity")
    
    # ========================================================================
    # Step 2: Define QoI and Sample Parameter Space
    # ========================================================================
    print("\n" + "="*70)
    print("Step 2: Sampling Parameter Space")
    print("="*70)
    
    def quantity_of_interest(params):
        """
        Evaluate QoI for given parameters.
        QoI = final state L2 norm (measure of solution magnitude at final time)
        """
        mu1, sigma1, mu2, sigma2, diffusivity, velocity = params
        
        # Create initial condition
        x = np.linspace(0, 1, N)
        u0 = (np.exp(-0.5 * ((x - mu1) / sigma1)**2) +
              0.5 * np.exp(-0.5 * ((x - mu2) / sigma2)**2))
        u0 = u0 / np.max(u0)  # Normalize
        
        # Create model with these parameters
        model = AdvectionDiffusion1D_CN(
            N=N, dt=dt, 
            diffusivity=max(diffusivity, 0.001),  # Avoid zero
            velocity=velocity
        )
        
        # Evolve
        trajectory = model.evolve(u0, n_steps=n_steps)
        
        # QoI: L2 norm at final time
        u_final = trajectory[-1, :]
        qoi = np.linalg.norm(u_final)
        
        return qoi
    
    def gradient_qoi_finite_diff(params, epsilon=1e-5):
        """
        Compute gradient of QoI using finite differences.
        """
        grad = np.zeros(n_params)
        qoi_center = quantity_of_interest(params)
        
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += epsilon
            qoi_plus = quantity_of_interest(params_plus)
            
            grad[i] = (qoi_plus - qoi_center) / epsilon
        
        return grad
    
    # Sample parameter space
    print("\nSampling parameter space...")
    n_samples = 100
    
    # Parameter bounds
    param_bounds = np.array([
        [0.2, 0.4],   # mu1
        [0.05, 0.15], # sigma1
        [0.6, 0.8],   # mu2
        [0.05, 0.15], # sigma2
        [0.005, 0.02],# diffusivity
        [0.3, 0.7]    # velocity
    ])
    
    # Latin Hypercube Sampling
    np.random.seed(42)
    samples = np.random.rand(n_samples, n_params)
    for i in range(n_params):
        samples[:, i] = (param_bounds[i, 0] + 
                        samples[:, i] * (param_bounds[i, 1] - param_bounds[i, 0]))
    
    print(f"Generated {n_samples} parameter samples")
    
    # Evaluate QoI and gradients
    print("Evaluating QoI and gradients...")
    qoi_values = np.zeros(n_samples)
    gradients = np.zeros((n_samples, n_params))
    
    for i in range(n_samples):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        qoi_values[i] = quantity_of_interest(samples[i])
        gradients[i] = gradient_qoi_finite_diff(samples[i])
    
    print(f"✓ Evaluation complete")
    print(f"  QoI range: [{qoi_values.min():.3f}, {qoi_values.max():.3f}]")
    
    # ========================================================================
    # Step 3: Build Active Subspace
    # ========================================================================
    print("\n" + "="*70)
    print("Step 3: Building Active Subspace")
    print("="*70)
    
    from boed.reduction.inference import ActiveSubspaces
    
    # Build AS
    as_rank = 2  # Focus on top 2 dimensions
    as_model = ActiveSubspaces(rank=as_rank)
    as_model.fit_from_gradients(samples, gradients)
    
    print(f"✓ Active Subspace built: rank = {as_rank}")
    print(f"\nEigenvalues (importance of each direction):")
    for i, eigval in enumerate(as_model.eigenvalues[:n_params]):
        pct = 100 * eigval / as_model.eigenvalues.sum()
        print(f"  λ_{i+1} = {eigval:.4f} ({pct:.1f}%)")
    
    # Eigenvectors show which parameters are important
    print(f"\nActive directions (eigenvectors):")
    W = as_model.basis  # (n_params, rank)
    
    param_names = ['mu1', 'sig1', 'mu2', 'sig2', 'diff', 'vel']
    
    for i in range(as_rank):
        print(f"\n  Direction {i+1} (λ = {as_model.eigenvalues[i]:.4f}):")
        for j, name in enumerate(param_names):
            print(f"    {name:6s}: {W[j, i]:+.3f}")
    
    # ========================================================================
    # Step 4: Analyze Active vs Inactive Subspace
    # ========================================================================
    print("\n" + "="*70)
    print("Step 4: Active vs Inactive Subspace Analysis")
    print("="*70)
    
    # Project samples onto active subspace
    samples_active = as_model.transform(samples)
    
    print(f"Projected samples: {samples.shape} → {samples_active.shape}")
    
    # Compute variance explained
    total_variance = np.sum(as_model.eigenvalues)
    active_variance = np.sum(as_model.eigenvalues[:as_rank])
    variance_ratio = active_variance / total_variance
    
    print(f"\nVariance in active subspace: {variance_ratio:.2%}")
    print(f"Dimension reduction: {n_params} → {as_rank}")
    
    # ========================================================================
    # Step 5: Implications for BOED
    # ========================================================================
    print("\n" + "="*70)
    print("Step 5: BOED Design Recommendations")
    print("="*70)
    
    # Identify most important parameters
    importance = np.sum(np.abs(W), axis=1)  # Sum over active directions
    importance = importance / importance.sum()
    
    print("\nParameter importance ranking:")
    sorted_idx = np.argsort(importance)[::-1]
    for i in sorted_idx:
        print(f"  {param_names[i]:6s}: {importance[i]:.1%} important")
    
    print("\n💡 BOED Design Strategy:")
    print(f"   Focus on parameters: {', '.join([param_names[i] for i in sorted_idx[:3]])}")
    print(f"   These explain {importance[sorted_idx[:3]].sum():.1%} of QoI variation")
    print(f"   Consider fixing less important parameters")
    
    print("\n📊 Computational Benefits:")
    print(f"   Full space BOED: {n_params} parameters")
    print(f"   Active subspace BOED: {as_rank} parameters")
    print(f"   Complexity reduction: ~{(n_params/as_rank)**2:.0f}x")
    
    # ========================================================================
    # Step 6: Visualization
    # ========================================================================
    print("\n" + "="*70)
    print("Step 6: Visualization")
    print("="*70)
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Eigenvalue decay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    eigenvalues = as_model.eigenvalues
    
    # Linear scale
    ax1.plot(range(1, n_params+1), eigenvalues, 'o-', 
             linewidth=2, markersize=8)
    ax1.axvline(as_rank + 0.5, color='r', linestyle='--', 
                label=f'Active rank = {as_rank}')
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Active Subspace Eigenvalues', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(range(1, n_params+1))
    
    # Cumulative variance
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    ax2.plot(range(1, n_params+1), cumulative, 'g-', 
             linewidth=2, marker='o', markersize=8)
    ax2.axvline(as_rank + 0.5, color='r', linestyle='--',
                label=f'Active rank = {as_rank}')
    ax2.axhline(variance_ratio, color='b', linestyle=':',
                label=f'{variance_ratio:.1%} variance')
    ax2.set_xlabel('Number of dimensions', fontsize=12)
    ax2.set_ylabel('Cumulative variance explained', fontsize=12)
    ax2.set_title('Variance Explained', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(range(1, n_params+1))
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'as_eigenvalues.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'as_eigenvalues.png'}")
    
    # Plot 2: Parameter importance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(importance / importance.max())
    bars = ax.barh(param_names, importance, color=colors)
    ax.set_xlabel('Relative Importance', fontsize=12)
    ax.set_title('Parameter Importance (Active Subspace)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val + 0.01, i, f'{val:.1%}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'as_parameter_importance.png', 
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'as_parameter_importance.png'}")
    
    # Plot 3: Active subspace projection
    if as_rank >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot in active subspace
        scatter = ax1.scatter(samples_active[:, 0], samples_active[:, 1],
                            c=qoi_values, cmap='viridis', s=50, alpha=0.7)
        ax1.set_xlabel('Active Direction 1', fontsize=12)
        ax1.set_ylabel('Active Direction 2', fontsize=12)
        ax1.set_title('QoI in Active Subspace', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='QoI value')
        
        # Gradient magnitude plot
        grad_mag = np.linalg.norm(gradients, axis=1)
        scatter2 = ax2.scatter(samples_active[:, 0], samples_active[:, 1],
                             c=grad_mag, cmap='plasma', s=50, alpha=0.7)
        ax2.set_xlabel('Active Direction 1', fontsize=12)
        ax2.set_ylabel('Active Direction 2', fontsize=12)
        ax2.set_title('Gradient Magnitude in Active Subspace', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='||∇QoI||')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'as_projection.png', 
                    dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'as_projection.png'}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    print(f"""
This example demonstrated:

1. ✓ Multi-parameter problem setup ({n_params} parameters)
2. ✓ QoI evaluation and gradient computation
3. ✓ Active Subspace identification (rank {as_rank})
4. ✓ Parameter importance ranking
5. ✓ Dimension reduction: {n_params} → {as_rank} ({variance_ratio:.1%} variance)

Key findings:
""")
    
    print(f"Most important parameters:")
    for i in sorted_idx[:3]:
        print(f"  - {param_names[i]}: {importance[i]:.1%}")
    
    print(f"""
Least important parameters:""")
    for i in sorted_idx[-2:]:
        print(f"  - {param_names[i]}: {importance[i]:.1%}")
    
    print(f"""
BOED Strategy:
- Focus sensors on monitoring {', '.join([param_names[i] for i in sorted_idx[:2]])}
- Consider fixing {', '.join([param_names[i] for i in sorted_idx[-2:]])}
- Expected complexity reduction: ~{(n_params/as_rank)**2:.0f}x
- Active subspace explains {variance_ratio:.1%} of QoI variation

Next steps:
1. Run BOED in {as_rank}D active subspace
2. Compare design quality vs full {n_params}D space
3. Validate parameter estimates on test data

Benefits:
    Active Subspace pre-screening identifies which parameters 
    actually matter BEFORE running expensive BOED, focusing 
    computational effort where it counts.
""")
    
    print("="*70)
    print("Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()
