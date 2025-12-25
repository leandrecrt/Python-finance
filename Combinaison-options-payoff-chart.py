import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Objectif
# - Tu rentres UNE FOIS les paramètres de marché (S0, r, q, sigma, T)
# - Ensuite, pour chaque leg option (call/put), la prime est PRICÉE en Black–Scholes
# - Donc tu ne peux plus "inventer" des primes qui créent un arbitrage évident.
# - Futures: F0 calculé via cost-of-carry (même T)
# - ZC bond: price_today calculé via exp(-rT) (même T)
# - Puis on trace le PnL à maturité: somme des PnL de chaque leg, point par point.
# ============================================================

# -------------------------
# Inputs robustes
# -------------------------
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

# -------------------------
# Normal CDF (sans scipy)
# -------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# -------------------------
# Black–Scholes (European, dividend yield q)
# -------------------------
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

# -------------------------
# Forward/Futures (cost of carry)
# -------------------------
def forward_price(S0: float, r: float, q: float, T: float) -> float:
    return S0 * math.exp((r - q) * T)

# -------------------------
# PnL à maturité (sur une grille S_T commune)
# -------------------------
def pnl_vanilla_option(S_T, position, option_type, K, premium):
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)
    return payoff - premium if position == "long" else premium - payoff

def pnl_futures(S_T, position, F0, qty):
    sign = 1.0 if position == "long" else -1.0
    return qty * sign * (S_T - F0)

def pnl_zcb(S_T, position, price_today, face_value):
    const = (face_value - price_today) if position == "long" else (price_today - face_value)
    return np.full_like(S_T, const, dtype=float)

# -------------------------
# Break-evens propres (uniquement crossings)
# -------------------------
def find_breakevens_crossings(x, y, eps=1e-12):
    y2 = y.copy()
    y2[np.abs(y2) < eps] = 0.0
    s = np.sign(y2)

    # Propagation du dernier signe non nul pour ignorer les plateaux à 0
    last = 0.0
    for i in range(len(s)):
        if s[i] == 0.0:
            s[i] = last
        else:
            last = s[i]

    bes = []
    for i in range(len(s) - 1):
        if s[i] == 0.0 or s[i+1] == 0.0:
            continue
        if s[i] != s[i+1]:
            x0, x1 = x[i], x[i+1]
            y0, y1 = y[i], y[i+1]
            if y1 != y0:
                x_star = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
                bes.append(x_star)

    bes = sorted(bes)
    cleaned = []
    for b in bes:
        if not cleaned or abs(b - cleaned[-1]) > 1e-6:
            cleaned.append(b)
    return cleaned

# ============================================================
# MAIN
# ============================================================
print("=== Combo PnL (Options + ZC Bond + Futures) avec PRICING au début ===\n")

print("Paramètres de marché (utilisés pour PRICER les primes + F0 + ZC)")
S0 = ask_float("Spot S0 : ")
r  = ask_float("Taux sans risque r (ex 0.03) : ", allow_negative=True)
q  = ask_float("Dividend yield / carry q (ex 0.01, 0 si aucun) : ", allow_negative=True)
sigma = ask_float("Volatilité sigma (ex 0.20) : ")
T  = ask_float("Maturité T en années (ex 0.5) : ")

if T <= 0:
    raise ValueError("T doit être > 0 pour ce script (pricing).")

F0_theo = forward_price(S0, r, q, T)
print(f"\n➡️ Futures/Forward théorique (même maturité T): F0 = {F0_theo:.6f}")
print("➡️ ZC bond price: price = face * exp(-rT)")

n = ask_int("\nNombre de legs ? 👉 ")
if n <= 0:
    raise ValueError("Nombre de legs doit être > 0")

legs = []
refs = [S0, F0_theo]  # pour auto-range de l'axe X

# ------------------------------------------------------------
# Saisie des legs
# ------------------------------------------------------------
for i in range(n):
    print(f"\n--- Leg {i+1}/{n} ---")
    pos_choice = ask_int("Position (1=Short, 2=Long) 👉 ", valid=[1, 2])
    position = "short" if pos_choice == 1 else "long"

    print("Instrument : 1=Call (BS) | 2=Put (BS) | 3=Zero-coupon bond (priced) | 4=Futures (priced)")
    inst = ask_int("👉 ", valid=[1, 2, 3, 4])

    if inst in (1, 2):
        option_type = "call" if inst == 1 else "put"
        K = ask_float("Strike K : ")
        premium = bs_price(S0, K, r, q, sigma, T, option_type)

        print(f"✅ Prime Black–Scholes {option_type.upper()}(K={K}) = {premium:.6f}")

        legs.append(("option", position, option_type, K, premium))
        refs.append(K)

    elif inst == 4:
        qty = ask_float("Quantité (ex 1) : ")
        # Futures priced => F0 fixé par modèle
        print(f"✅ Futures price F0 (modèle) = {F0_theo:.6f}")
        legs.append(("futures", position, F0_theo, qty))
        refs.append(F0_theo)

    else:  # ZC bond
        face = ask_float("Valeur à maturité (Face value) : ")
        price_today = face * math.exp(-r * T)
        print(f"✅ ZC bond price aujourd'hui = {price_today:.6f} (pour face={face})")
        legs.append(("zcb", position, price_today, face))
        refs.append(face)

# ------------------------------------------------------------
# Grille S_T AUTOMATIQUE (pas de question)
# ------------------------------------------------------------
min_ref = min(refs)
max_ref = max(refs)
spread = max(max_ref - min_ref, max_ref * 0.5, 10.0)

S_min = 0.0
S_max = max_ref + spread

nb_points = int(np.clip((S_max - S_min) * 10, 1500, 25000))
S_T = np.linspace(S_min, S_max, nb_points)

# ------------------------------------------------------------
# Somme des PnL point par point
# ------------------------------------------------------------
pnl_total = np.zeros_like(S_T, dtype=float)

for leg in legs:
    kind = leg[0]
    if kind == "option":
        _, position, option_type, K, premium = leg
        pnl_total += pnl_vanilla_option(S_T, position, option_type, K, premium)
    elif kind == "futures":
        _, position, F0, qty = leg
        pnl_total += pnl_futures(S_T, position, F0, qty)
    elif kind == "zcb":
        _, position, price_today, face = leg
        pnl_total += pnl_zcb(S_T, position, price_today, face)

# ------------------------------------------------------------
# Break-evens + plot vert/rouge
# ------------------------------------------------------------
breakevens = find_breakevens_crossings(S_T, pnl_total)

pnl_green = np.where(pnl_total >= 0, pnl_total, np.nan)
pnl_red   = np.where(pnl_total < 0, pnl_total, np.nan)

plt.figure(figsize=(10, 6))
plt.plot(S_T, pnl_green, lw=2, color="green", label="PnL >= 0")
plt.plot(S_T, pnl_red,   lw=2, color="red",   label="PnL < 0")
plt.axhline(0, color="black", ls="--", lw=1)

# Break-evens (affiche max 8 pour rester propre)
for b in breakevens[:8]:
    plt.axvline(b, color="grey", ls=":", lw=1)

be_txt = ", ".join([f"{b:.2f}" for b in breakevens[:8]])
if len(breakevens) > 8:
    be_txt += f" ... (+{len(breakevens)-8})"

plt.title(f"PnL total (primes pricées BS) — BE: {be_txt if breakevens else 'aucun'}")
plt.xlabel("Prix du sous-jacent à maturité (S_T)")
plt.ylabel("Profit / Loss")
plt.grid(True)
plt.legend()

plt.savefig("pnl_combo_priced.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Fichier: pnl_combo_priced.png")
print("Break-even(s):", [round(b, 4) for b in breakevens] if breakevens else "aucun")

# Note importante (sans bloquer):
print("\nNote:")
print("- Les primes sont 'cohérentes no-arbitrage' DANS le modèle Black–Scholes (European, q continu).")
print("- En marché réel, bid/ask, smile/skew, dividends discrets, etc. peuvent créer des écarts.")

