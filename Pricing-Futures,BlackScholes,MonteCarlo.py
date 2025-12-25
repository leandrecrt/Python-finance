import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# INPUTS
# =========================
def ask_int(prompt, valid=None):
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if valid is None or v in valid:
                return v
        except ValueError:
            pass
        print("❌ Entrée invalide.")

def ask_float(prompt, allow_negative=False):
    while True:
        s = input(prompt).strip().replace(",", ".")
        try:
            v = float(s)
            if allow_negative or v >= 0:
                return v
        except ValueError:
            pass
        print("❌ Mets un nombre valide.")

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# =========================
# FORWARD / FUTURES (cost of carry)
# =========================
def forward_futures_price(S0: float, r: float, q: float, T: float) -> float:
    # Modèle simple (taux déterministes) : Forward ≈ Futures
    return S0 * math.exp((r - q) * T)

# =========================
# BLACK-SCHOLES (European)
# =========================
def bs_price(S0: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str) -> float:
    if T <= 0:
        return max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)

    if sigma <= 0:
        fwd = S0 * math.exp((r - q) * T)
        disc = math.exp(-r * T)
        return disc * max(fwd - K, 0.0) if option_type == "call" else disc * max(K - fwd, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        return S0 * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    else:
        return K * disc_r * norm_cdf(-d2) - S0 * disc_q * norm_cdf(-d1)

# =========================
# MONTE CARLO: paths + option price
# =========================
def simulate_gbm_paths(S0, r, q, sigma, T, n_steps, n_sims, seed=42):
    """
    GBM sous mesure risque-neutre:
    dS/S = (r-q)dt + sigma dW
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)

    Z = rng.standard_normal((n_sims, n_steps))
    increments = drift + vol * Z

    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_sims, 1)), log_paths])
    paths = S0 * np.exp(log_paths)
    return paths

def mc_option_price_from_paths(paths, K, r, T, option_type):
    ST = paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc = math.exp(-r * T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / math.sqrt(len(payoff))
    return float(price), float(stderr)

# =========================
# MAIN
# =========================
print("=== Pricing Tool : Forward/Futures & Options (BS / Monte Carlo) ===\n")
print("1 = Forward / Futures")
print("2 = Options (European)")
main_choice = ask_int("👉 Choix : ", valid=[1, 2])

# -------------------------------------------------
# 1) Forward / Futures : courbe T -> F0(T)
# -------------------------------------------------
if main_choice == 1:
    print("\n--- Forward / Futures ---")
    print("1 = Forward")
    print("2 = Futures")
    ff_choice = ask_int("👉 ", valid=[1, 2])
    label = "Forward" if ff_choice == 1 else "Futures"

    S0 = ask_float("Spot S0 : ")
    r = ask_float("Taux sans risque r (ex 0.03) : ", allow_negative=True)
    q = ask_float("Carry / dividend yield q (ex 0.01, 0 si aucun) : ", allow_negative=True)

    # Prix à une maturité donnée
    T0 = ask_float("Maturité T0 (années) pour calculer un prix (ex 0.5) : ")
    F0_T0 = forward_futures_price(S0, r, q, T0)
    print(f"\n✅ {label} théorique à T0={T0} : F0(T0) = {F0_T0:.6f}")
    print("Formule: F0(T) = S0 * exp((r - q) * T)")

    # Courbe demandée : x = maturité, y = valeur du futures/forward
    T_max = ask_float("\nT max pour la courbe (années, ex 2 ou 5) : ")
    if T_max <= 0:
        raise ValueError("T max doit être > 0.")

    T_grid = np.linspace(0.0, T_max, 500)
    F_grid = S0 * np.exp((r - q) * T_grid)

    plt.figure(figsize=(9, 5))
    plt.plot(T_grid, F_grid, linewidth=2, label=f"{label} théorique F0(T)")
    plt.scatter([T0], [F0_T0], label=f"Point à T0={T0}", zorder=3)
    plt.title(f"{label} : courbe maturité T (x) vs prix F0(T) (y)")
    plt.xlabel("Maturité T (années)")
    plt.ylabel("Prix Forward/Futures F0(T)")
    plt.grid(True)
    plt.legend()
    plt.show()

# -------------------------------------------------
# 2) Options : BS ou MC + courbes
# -------------------------------------------------
else:
    print("\n--- Options (European) ---")
    print("Type : 1 = Call | 2 = Put")
    opt_choice = ask_int("👉 ", valid=[1, 2])
    option_type = "call" if opt_choice == 1 else "put"

    print("\nMéthode : 1 = Black–Scholes | 2 = Monte Carlo")
    method = ask_int("👉 ", valid=[1, 2])

    S0 = ask_float("Spot S0 : ")
    K = ask_float("Strike K : ")
    r = ask_float("Taux sans risque r (ex 0.03) : ", allow_negative=True)
    q = ask_float("Dividend yield q (ex 0.01, 0 si aucun) : ", allow_negative=True)
    sigma = ask_float("Volatilité sigma (ex 0.20) : ")
    T = ask_float("Maturité T (années, ex 0.5) : ")

    if method == 1:
        price = bs_price(S0, K, r, q, sigma, T, option_type)
        print(f"\n✅ Black–Scholes {option_type.upper()} = {price:.6f}")

        # Courbe prime vs S0
        S_min = max(0.0, 0.5 * S0)
        S_max = 1.5 * S0 if S0 > 0 else 200.0
        S_grid = np.linspace(S_min, S_max, 300)
        prices = np.array([bs_price(s, K, r, q, sigma, T, option_type) for s in S_grid])

        plt.figure(figsize=(9, 5))
        plt.plot(S_grid, prices, linewidth=2)
        plt.scatter([S0], [price], zorder=3)
        plt.title(f"Black–Scholes : prime {option_type.upper()} (y) vs S0 (x)")
        plt.xlabel("Sous-jacent S0")
        plt.ylabel("Prime")
        plt.grid(True)
        plt.show()

    else:
        n_sims = ask_int("Nombre de simulations (ex 20000) : ")
        n_steps = ask_int("Nombre de pas de temps (ex 252) : ")
        seed = ask_int("Seed (ex 42) : ")

        paths = simulate_gbm_paths(S0, r, q, sigma, T, n_steps, n_sims, seed=seed)
        price, stderr = mc_option_price_from_paths(paths, K, r, T, option_type)

        print(f"\n✅ Monte Carlo {option_type.upper()} = {price:.6f}  (stderr ≈ {stderr:.6f})")
        print("Info: n_steps=granularité du chemin (252 ~ jours de bourse/an), seed=répétabilité des tirages.")

        # Graph trajectoires + moyenne
        t = np.linspace(0, T, n_steps + 1)
        plt.figure(figsize=(10, 6))

        max_show = min(200, n_sims)
        plt.plot(t, paths[:max_show].T, linewidth=0.6, alpha=0.25)

        mean_path = paths.mean(axis=0)
        plt.plot(t, mean_path, linewidth=2.5)

        plt.title("Monte Carlo GBM : trajectoires (gris) + moyenne (épaisse)")
        plt.xlabel("Temps (années)")
        plt.ylabel("Prix du sous-jacent")
        plt.grid(True)
        plt.show()