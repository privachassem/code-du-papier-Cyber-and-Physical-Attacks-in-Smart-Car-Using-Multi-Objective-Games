import numpy as np
import matplotlib.pyplot as plt

# Base parameters: Costs and Gains for Defender (Dc) and Attacker (Ac)
# Cyber Layer
C1_Dc, C2_Dc, C1_Ac, C2_Ac = 3, 4, 5, 6
G1_Dc, G2_Dc, G1_Ac, G2_Ac = 7, 8, 5, 6

# Physical Layer
C1_Dp, C2_Dp, C1_Ap, C2_Ap = 2, 3, 4, 5
G1_Dp, G2_Dp, G1_Ap, G2_Ap = 6, 7, 4, 5

# Probabilities of propagation for indirect attacks
theta_values = np.linspace(0, 1, 10)  # Cyber -> Physical attack propagation
lambda_values = np.linspace(0, 1, 10)  # Physical -> Cyber attack propagation

# Different weights to test for Pareto-Nash approach
weights = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

# Payoff functions for indirect and direct attacks
def payoff_indirect_cyber(theta, weight_direct, weight_indirect):
    U_Dc1 = -C1_Dc
    U_Dc2 = G2_Dc - theta * (G1_Ap + C2_Dc)
    return weight_direct * U_Dc1 + weight_indirect * U_Dc2

def payoff_direct_physical(lambda_, weight_direct, weight_indirect):
    U_Dp1 = -C1_Dp
    U_Dp2 = G2_Dp - lambda_ * (G1_Ac + C2_Dp)
    return weight_direct * U_Dp1 + weight_indirect * U_Dp2

# Greedy and Random strategy payoffs for comparison
def greedy_payoff(theta, lambda_):
    return max(payoff_indirect_cyber(theta, 1, 0), payoff_direct_physical(lambda_, 1, 0))

def random1_payoff(theta, lambda_):
    # Randomly select a payoff from either the cyber layer or the physical layer
    if np.random.rand() > 0.5:
        # Cyber layer (indirect attack)
        return np.random.uniform(payoff_indirect_cyber(theta, 1, 0), payoff_direct_physical(lambda_, 1, 0))
    else:
        # Physical layer (direct attack)
        return np.random.uniform(payoff_indirect_cyber(theta, 1, 0), payoff_direct_physical(lambda_, 1, 0))
        
        
# Visualization
fig, axs = plt.subplots(2, len(weights), figsize=(16, 10))
fig.suptitle("Comparison of Pareto-Nash, Greedy, and Random Strategies")

for idx, (weight_direct, weight_indirect) in enumerate(weights):
    pareto_nash_cyber = [payoff_indirect_cyber(theta, weight_direct, weight_indirect) for theta in theta_values]
    pareto_nash_physical = [payoff_direct_physical(lambda_, weight_direct, weight_indirect) for lambda_ in lambda_values]
    greedy_cyber = [greedy_payoff(theta, 0) for theta in theta_values]
    greedy_physical = [greedy_payoff(0, lambda_) for lambda_ in lambda_values]
    random_physical = [random1_payoff(0, lambda_) for lambda_ in lambda_values]
    random_cyber = [random1_payoff(theta, 0) for theta in theta_values]

    # Plot Cyber Layer (varying theta)
    axs[0, idx].plot(theta_values, pareto_nash_cyber, label="Pareto-Nash", color="blue")
    axs[0, idx].plot(theta_values, greedy_cyber, label="Greedy", color="green")
    axs[0, idx].plot(theta_values, random_cyber, label="Random", color="orange")
    axs[0, idx].set_title(f"Cyber Layer (Weights: {weight_direct}, {weight_indirect})")
    axs[0, idx].set_xlabel("Theta")
    axs[0, idx].set_ylabel("Payoff")
    axs[0, idx].legend()

    # Plot Physical Layer (varying lambda)
    axs[1, idx].plot(lambda_values, pareto_nash_physical, label="Pareto-Nash", color="blue")
    axs[1, idx].plot(lambda_values, greedy_physical, label="Greedy", color="green")
    axs[1, idx].plot(lambda_values, random_physical, label="Random", color="orange")
    axs[1, idx].set_title(f"Physical Layer (Weights: {weight_direct}, {weight_indirect})")
    axs[1, idx].set_xlabel("Lambda")
    axs[1, idx].set_ylabel("Payoff")
    axs[1, idx].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
