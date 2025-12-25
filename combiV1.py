import numpy as np
import matplotlib.pyplot as plt

# =========================
# INPUTS ROBUSTES
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
        print("❌ Mets un nombre (ex: 100 ou 100.5).")

# =========================
# PAYOFFS / PnL PAR LEG (SUR UNE GRILLE COMMUNE)
# =========================
def pnl_vanilla_option(S_T, position, option_type, K, premium):
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:  # put
        payoff = np.maximum(K - S_T, 0.0)
    return payoff - premium if position == "long" else premium - payoff

def pnl_digital(S_T, position, digital_type, K, premium, payout):
    # payout * 1{ITM}
    if digital_type == "digital_call":
        payoff = payout * (S_T >= K).astype(float)
    else:  # digital_put
        payoff = payout * (S_T <= K).astype(float)
    return payoff - premium if position == "long" else premium - payoff

def pnl_zcb(S_T, position, price_today, face_value):
    const = (face_value - price_today) if position == "long" else (price_today - face_value)
    return np.full_like(S_T, const, dtype=float)

def pnl_futures(S_T, position, F0, qty=1.0):
    sign = 1.0 if position == "long" else -1.0
    return qty * sign * (S_T - F0)

# =========================
# BREAK-EVENS (PROPRE : UNIQUEMENT TRAVERSÉES DE 0)
# =========================
def find_breakevens_crossings(x, y, eps=1e-12):
    """
    Break-even = endroits où y traverse 0 (changement de signe).
    On ignore les plateaux à 0 (sinon digital => spam).
    """
    y2 = y.copy()
    y2[np.abs(y2) < eps] = 0.0

    # signe : -1, 0, +1
    s = np.sign(y2)

    # Pour éviter les 0 qui cassent le test, on "propage" le dernier signe non nul
    # (ça supprime les plateaux exacts à 0)
    last = 0
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = last
        else:
            last = s[i]

    bes = []
    for i in range(len(s) - 1):
        if s[i] == 0 or s[i+1] == 0:
            continue
        if s[i] != s[i+1]:
            # interpolation linéaire
            x0, x1 = x[i], x[i+1]
            y0, y1 = y[i], y[i+1]
            if y1 != y0:
                x_star = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                bes.append(x_star)

    # clean / dédoublonnage
    bes = sorted(bes)
    cleaned = []
    for b in bes:
        if not cleaned or abs(b - cleaned[-1]) > 1e-6:
            cleaned.append(b)
    return cleaned

# =========================
# MAIN
# =========================
print("=== PnL TOTAL — SOMME DES LEGS (point par point) ===")
n = ask_int("Nombre de legs ? 👉 ")
legs = []
refs = []
has_digital = False

for i in range(n):
    print(f"\n--- Leg {i+1}/{n} ---")

    pos_choice = ask_int("Position (1=Short, 2=Long) 👉 ", valid=[1, 2])
    position = "short" if pos_choice == 1 else "long"

    print("Instrument :")
    print("1 = Call (vanilla)")
    print("2 = Put (vanilla)")
    print("3 = Zero-coupon bond")
    print("4 = Futures")
    print("5 = Digital Call")
    print("6 = Digital Put")
    inst_choice = ask_int("👉 ", valid=[1, 2, 3, 4, 5, 6])

    if inst_choice in (1, 2):
        option_type = "call" if inst_choice == 1 else "put"
        K = ask_float("Strike K : ")
        premium = ask_float("Prime : ")
        legs.append(("vanilla", position, option_type, K, premium))
        refs.append(K)

    elif inst_choice == 3:
        price_today = ask_float("Prix aujourd'hui : ")
        face_value = ask_float("Valeur à maturité : ")
        legs.append(("zcb", position, price_today, face_value))
        refs.append(face_value)

    elif inst_choice == 4:
        F0 = ask_float("Prix futures F0 : ")
        qty = ask_float("Quantité : ")
        legs.append(("futures", position, F0, qty))
        refs.append(F0)

    else:
        has_digital = True
        digital_type = "digital_call" if inst_choice == 5 else "digital_put"
        K = ask_float("Strike K : ")
        payout = ask_float("Payout (montant payé si ITM, ex: 1) : ")
        premium = ask_float("Prime : ")
        legs.append(("digital", position, digital_type, K, premium, payout))
        refs.append(K)

# =========================
# AXE S_T AUTO (PAS DE QUESTION)
# =========================
min_ref = min(refs) if refs else 0.0
max_ref = max(refs) if refs else 100.0
spread = max(max_ref - min_ref, max_ref * 0.5, 10.0)

S_min = 0.0
S_max = max_ref + spread

# plus de points si range large
nb_points = int(np.clip((S_max - S_min) * 10, 1500, 25000))
S_T = np.linspace(S_min, S_max, nb_points)

# =========================
# SOMME DES PnL (POINT PAR POINT)
# =========================
pnl_total = np.zeros_like(S_T, dtype=float)

for leg in legs:
    kind = leg[0]

    if kind == "vanilla":
        _, position, option_type, K, premium = leg
        pnl_total += pnl_vanilla_option(S_T, position, option_type, K, premium)

    elif kind == "digital":
        _, position, digital_type, K, premium, payout = leg
        pnl_total += pnl_digital(S_T, position, digital_type, K, premium, payout)

    elif kind == "zcb":
        _, position, price_today, face_value = leg
        pnl_total += pnl_zcb(S_T, position, price_today, face_value)

    elif kind == "futures":
        _, position, F0, qty = leg
        pnl_total += pnl_futures(S_T, position, F0, qty)

# =========================
# BREAK-EVENS + GRAPH CLEAN
# =========================
breakevens = find_breakevens_crossings(S_T, pnl_total)

pnl_green = np.where(pnl_total >= 0, pnl_total, np.nan)
pnl_red   = np.where(pnl_total < 0, pnl_total, np.nan)

plt.figure(figsize=(10, 6))

# Si digital présent -> rendu en escaliers plus propre
drawstyle = "steps-post" if has_digital else "default"

plt.plot(S_T, pnl_green, linewidth=2, color="green", label="PnL >= 0", drawstyle=drawstyle)
plt.plot(S_T, pnl_red,   linewidth=2, color="red",   label="PnL < 0",  drawstyle=drawstyle)

plt.axhline(0, color="black", linestyle="--", linewidth=1)

# break-evens (on en affiche max 6 pour éviter le spam visuel)
for b in breakevens[:6]:
    plt.axvline(b, color="grey", linestyle=":", linewidth=1)

be_txt = ", ".join([f"{b:.2f}" for b in breakevens[:6]])
if len(breakevens) > 6:
    be_txt += f" ... (+{len(breakevens)-6})"

plt.title(f"PnL total (combo) — BE: {be_txt if breakevens else 'aucun'}")
plt.xlabel("Prix du sous-jacent à maturité (S_T)")
plt.ylabel("Profit / Loss")
plt.grid(True)
plt.legend()

plt.savefig("pnl_combo.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Graph sauvegardé : pnl_combo.png")
print("Break-even(s) :", [round(b, 4) for b in breakevens] if breakevens else "aucun")
