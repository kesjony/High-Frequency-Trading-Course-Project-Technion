import numpy as np
from numpy import sqrt, exp, sinh, cosh
import matplotlib.pyplot as plt

### ------------------------------- Problem 1 ---------------------------------------
sigma = 0.1
alpha = 1
phi = 0.11
k = 0.5
b = 0.1
T = 10
X0 = 10
S0 = 10
R = 10
N = 1000

t = np.linspace(0, T, N)

gamma = sqrt(phi / k)

def Q_for_a(alpha=0.1):
    zetta = (alpha - 0.5 * b + sqrt(phi * k)) / (alpha - 0.5 * b - sqrt(phi * k))
    v = gamma * (zetta * exp(gamma * (T - t)) + exp(-gamma * (T - t))) / (zetta * exp(gamma * T) + exp(-gamma * T)) * R
    Q = (zetta * exp(gamma * (T - t)) - exp(-gamma * (T - t))) / (zetta * exp(gamma * T) + exp(-gamma * T)) * R
    return Q, v

alphas = [0.01, 0.2, 0.5, 1, 10, 100, 1000]
colors = plt.get_cmap('autumn')(np.linspace(0.2, 0.7, len(alphas)))
vs = []
for i, a in enumerate(alphas):
    q, v = Q_for_a(a)
    plt.plot(t, q, label=f'a={a}', color=colors[-i-1])
    vs.append(v)

# The limit of alpha -> inf
Q_lim = (sinh(gamma * (T - t))) / (sinh(gamma * T)) * R
plt.plot(t, Q_lim, label=f'a->inf')

plt.xlabel('t')
plt.ylabel('Quantity')
plt.legend()
plt.savefig('Simulations_Plots/problem_1_quantity.png')
plt.xlim((5, 10.1))
plt.ylim((0, 1))
plt.savefig('Simulations_Plots/problem_1_quantity_zoom.png')
plt.clf()

for i, a in enumerate(alphas):
    plt.plot(t, vs[i], label=f'a={a}', color=colors[-i-1])

# The limit of alpha -> inf
v_lim = gamma * (cosh(gamma * (T - t))) / (sinh(gamma * T)) * R
plt.plot(t, v_lim, label=f'a->inf')

plt.xlabel('t')
plt.ylabel('Trading speed')
plt.legend()
plt.savefig('Simulations_Plots/problem_1_trading_speed.png')
plt.xlim((5, 10.1))
plt.ylim((-0.05, 0.6))
plt.savefig('Simulations_Plots/problem_1_trading_speed_zoom.png')
plt.clf()


### ------------------------------- Problem 2 ---------------------------------------
sigma = 0.5
alpha = 100
phi = 5
k = 0.5
b = 3.6
T = 1
R = 20
N = 1000
kappa = 20       # Speed of mean reversion
dt = T / N        # Time step
lambda_e = 0.02    # Exponential eta parameter
lambda_p = 50.    # Poisson lambda parameter
xi = lambda_e * lambda_p
mu0 = (xi / kappa)        # Initial value
rho = 0.05         # fraction of the trade


# Generate the OU process
def ou_process():
    mu = np.zeros(N+1)
    mu[0] = mu0
    N_poisson = np.random.poisson(lambda_p*dt, N+2)
    dN = N_poisson
    eta = np.random.exponential(1. / lambda_e, size=N+1)
    for t in range(1, N+1):
        mu[t] = mu[t-1] - kappa * mu[t-1] * dt + eta[t-1] * dN[t]
    return np.array(mu)

def realization():
    mu_p = ou_process()
    mu_m = ou_process()

    zeta = (k + phi) / (alpha - 0.5 * b)

    Q = np.zeros(N + 1)
    Q[0] = R

    v = np.zeros(N)
    for t in range(N):
        tau = T - t * dt
        # Our
        v[t] = Q[t] / (tau + zeta) + \
               phi / (k + phi) * rho * (mu_m[t] - ((1 - exp(-kappa * tau)) * (mu_m[t] - xi / kappa) + xi * tau / kappa) / (kappa *(tau + zeta))) \
               - b * (mu_p[t] - mu_m[t]) / (2 * (k + phi) * (tau + zeta)) * ((1 - kappa * zeta) * exp(-kappa * tau) + kappa * (tau + zeta) - 1) / kappa**2

        # Book's
        # v[t] = Q[t] / (tau + zeta) + \
        #        phi / (k + phi) * rho * (mu_m[t] - ((1 - exp(-kappa * tau)) * (mu_m[t] - xi / kappa) + xi * tau / kappa) / (kappa *(tau + zeta))) \
        #        + b * (mu_p[t] - mu_m[t]) / ((k + phi) * (tau + zeta)) * ((1 - kappa * zeta) * exp(-kappa * tau) + kappa * (tau + zeta) - 1) / kappa**2

        Q[t+1] = Q[t] - v[t] * dt

    return mu_m[:N], v[:N] - rho * mu_m[:N], v[:N], Q[:N]


realizations = 3
mu_ms,  v_rho_mu_ms, vs, Qs = [], [], [], []
for i in range(realizations):
   mu_m, v_rho_mu_m, v, Q = realization()
   mu_ms.append(mu_m), v_rho_mu_ms.append(v_rho_mu_m), vs.append(v), Qs.append(Q)


t = np.linspace(0, T, N)

for i in range(realizations):
    plt.plot(t, mu_ms[i], label=f'realizaiton {i}')

plt.legend()
plt.xlabel('t')
plt.ylabel("Market's Selling Rate")
plt.savefig('Simulations_Plots/problem_2_mu_m.png')
plt.clf()

for i in range(realizations):
    plt.plot(t, v_rho_mu_ms[i], label=f'realizaiton {i}')

plt.legend()
plt.xlabel('t')
plt.ylabel("v - rho * mu_m")
plt.savefig('Simulations_Plots/problem_2_v_rho_mu_m.png')
plt.clf()

for i in range(realizations):
    plt.plot(t, Qs[i], label=f'realizaiton {i}')

plt.plot(t, R * (1 - t/T), '--', color='black', label='TWAP')
plt.legend()
plt.xlabel('t')
plt.ylabel('Inventory Q')
plt.savefig('Simulations_Plots/problem_2_quantity.png')
plt.clf()


for i in range(realizations):
    plt.plot(t, vs[i], label=f'realizaiton {i}')

plt.legend()
plt.xlabel('t')
plt.ylabel('Optimal Trading Rate v')
plt.savefig('Simulations_Plots/problem_2_speed.png')
plt.clf()

for i in range(realizations):
    plt.plot(t, vs[i], label=f'v^*')
    plt.plot(t, mu_ms[i], label=f'mu_m')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Optimal Trading Rate v')
    # plt.show()
    plt.savefig(f'Simulations_Plots/problem_2_realization_{i}.png')
    plt.clf()


### simulating a lot of realization in order to study some summarization about the model


# Generate the OU process
def ou_process():
    mu = np.zeros(N+1)
    mu[0] = mu0
    N_poisson = np.random.poisson(lambda_p * dt, N+2)
    dN = N_poisson
    eta = np.random.exponential(1. / lambda_e, size=N+1)
    for t in range(1, N+1):
        mu[t] = mu[t-1] - kappa * mu[t-1] * dt + eta[t-1] * dN[t]
    return np.array(mu)

# Simulation of realizations

def realization():
    mu_p = ou_process()
    mu_m = ou_process()
    #plt.plot(mu_m)
    #plt.show()
    zeta = (k + phi) / (alpha - 0.5 * b)

    Q = np.zeros(N + 1)
    Q[0] = R

    v = np.zeros(N)

    for t in range(N):
        tau = T - t * dt
        v[t] = Q[t] / (tau + zeta) + \
               phi / (k + phi) * rho * (mu_m[t] - ((1 - np.exp(-kappa * tau)) * (mu_m[t] - xi / kappa) + xi * tau / kappa) / (kappa * (tau + zeta))) \
               - b * (mu_p[t] - mu_m[t]) / (2 * (k + phi) * (tau + zeta)) * ((1 - kappa * zeta) * np.exp(-kappa * tau) + kappa * (tau + zeta) - 1) / kappa**2

        Q[t+1] = Q[t] - v[t] * dt

    return mu_m[:N], v[:N], Q[:N]

# Simulating 1000 realizations
realizations = 1000
mu_ms, vs, Qs = [], [], []
for i in range(realizations):
    mu_m, v, Q = realization()
    mu_ms.append(mu_m)
    vs.append(v)
    Qs.append(Q)

mu_ms = np.array(mu_ms)
vs = np.array(vs)
Qs = np.array(Qs)

# Compute mean, 75th percentile, and 25th percentile
mu_m_mean = np.mean(mu_ms, axis=0)
mu_m_75 = np.percentile(mu_ms, 75, axis=0)
mu_m_25 = np.percentile(mu_ms, 25, axis=0)

v_mean = np.mean(vs, axis=0)
v_75 = np.percentile(vs, 75, axis=0)
v_25 = np.percentile(vs, 25, axis=0)

Q_mean = np.mean(Qs, axis=0)
Q_75 = np.percentile(Qs, 75, axis=0)
Q_25 = np.percentile(Qs, 25, axis=0)

# Time axis
t = np.linspace(0, T, N)

# Plot for Q
plt.plot(t, Q_mean, label="Mean of Q")
plt.plot(t, Q_75, label="75th Percentile of Q", linestyle="--")
plt.plot(t, Q_25, label="25th Percentile of Q", linestyle="--")
plt.plot(t, R * (1 - t/T), '--', color='black', label='TWAP')
plt.fill_between(t, Q_25, Q_75, alpha=0.3)
plt.xlabel('t')
plt.ylabel('Inventory Q')
plt.legend()
plt.title("Mean, 75th and 25th Percentile of Inventory Q")
plt.savefig('Simulations_Plots/problem_2_inventory_1000_realizations.png')
plt.clf()

# Plot for v
plt.plot(t, v_mean, label="Mean of v")
plt.plot(t, v_75, label="75th Percentile of v", linestyle="--")
plt.plot(t, v_25, label="25th Percentile of v", linestyle="--")
plt.fill_between(t, v_25, v_75, alpha=0.3)
plt.xlabel('t')
plt.ylabel('Optimal Trading Rate v')
plt.legend()
plt.title("Mean, 75th and 25th Percentile of Trading Rate v")
plt.savefig('Simulations_Plots/problem_2_speed_1000_realizations.png')
plt.clf()

# Plot for mu_m
plt.plot(t, mu_m_mean, label="Mean of mu_m")
plt.plot(t, mu_m_75, label="75th Percentile of mu_m", linestyle="--")
plt.plot(t, mu_m_25, label="25th Percentile of mu_m", linestyle="--")
plt.fill_between(t, mu_m_25, mu_m_75, alpha=0.3)
plt.xlabel('t')
plt.ylabel("Market's Selling Rate mu_m")
plt.legend()
plt.title("Mean, 75th and 25th Percentile of Market's Selling Rate mu_m")
plt.savefig('Simulations_Plots/problem_2_Total_Sell_Volume_1000_realizations.png')
plt.clf()

