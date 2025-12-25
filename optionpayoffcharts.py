# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 21:42:40 2025

@author: leand
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# DIALOGUE UTILISATEUR
# =========================

print("Choisis la position :")
print("1 = Short")
print("2 = Long")
pos_choice = int(input("👉 Ton choix : "))

print("\nChoisis le type d'option :")
print("1 = Call")
print("2 = Put")
opt_choice = int(input("👉 Ton choix : "))

K = float(input("\nEntre le strike K : "))
premium = float(input("Entre la prime : "))

# =========================
# INTERPRÉTATION DES CHOIX
# =========================

position = "short" if pos_choice == 1 else "long"
option_type = "call" if opt_choice == 1 else "put"

# =========================
# PRIX DU SOUS-JACENT
# =========================

S_T = np.linspace(0, 2 * K, 500)

# =========================
# PAYOFF
# =========================

if option_type == "call":
    payoff = np.maximum(S_T - K, 0)
else:  # put
    payoff = np.maximum(K - S_T, 0)

# =========================
# PnL
# =========================

if position == "long":
    pnl = payoff - premium
else:  # short
    pnl = premium - payoff

# =========================
# GRAPH
# =========================

plt.figure(figsize=(9, 5))
plt.plot(S_T, pnl, linewidth=2)

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axvline(K, color="grey", linestyle="--", linewidth=1)

plt.title(f"PnL {position.capitalize()} {option_type.capitalize()} | K={K}, Prime={premium}")
plt.xlabel("Prix du sous-jacent à maturité $S_T$")
plt.ylabel("Profit / Loss")

plt.grid(True)
plt.show()
