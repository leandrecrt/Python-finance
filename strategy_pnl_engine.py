# strategy_pnl_engine.py
import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Glossary (expanded words)
# =========================
# S0 : Spot price (current underlying price) / Prix spot (prix actuel du sous-jacent)
# K  : Strike price / Prix d'exercice
# r  : Risk-free interest rate (annualized, continuously compounded) / Taux sans risque (annualisé, en continu)
# q  : Dividend yield / carry (annualized, continuously compounded) / Rendement du dividende / coût de portage (annualisé, en continu)
# sigma : Volatility (annualized) / Volatilité (annualisée)
# T  : Time to maturity (in years) / Temps jusqu'à maturité (en années)
# F0 : Forward/Futures theoretical price / Prix forward/future théorique

# ---------- Robust input helpers ----------
def ask_int(prompt, valid=None, min_value=None):
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if min_value is not None and v < min_value:
                raise ValueError
            if valid is None or v in valid:
                return v
        except ValueError:
            pass
        msg = "❌ Invalid input."
        if valid is not None:
            msg += f" Allowed: {sorted(valid)}"
        if min_value is not None:
            msg += f" | min={min_value}"
        print(msg)

def ask_float(prompt, allow_negative=False):
    while True:
        s = input(prompt).strip().replace(",", ".")
        try:
            v = float(s)
            if allow_negative or v >= 0:
                return v
        except ValueError:
            pass
        print("❌ Enter a valid number.")

def ask_percent(prompt, allow_negative=False, default=None):
    s = input(prompt).strip()
    if s == "" and default is not None:
        return float(default) / 100.0
    v = float(s.replace(",", "."))
    if not allow_negative and v < 0:
        raise ValueError("Percent cannot be negative here.")
    return v / 100.0

# ---------- Finance basics ----------
def forward_price(S0: float, r: float, q: float, T: float) -> float:
    # Cost-of-carry forward
    return S0 * math.exp((r - q) * T)

# ---------- PnL at maturity ----------
def pnl_option_at_maturity(ST: np.ndarray, option_type: str, K: float, premium: float, position: str, qty: float):
    option_type = option_type.lower()
    position = position.lower()
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    sign = 1.0 if position == "long" else -1.0
    # PnL = sign * qty * (payoff - premium)
    return sign * qty * (payoff - premium)

def pnl_futures_at_maturity(ST: np.ndarray, F0: float, position: str, qty: float):
    position = position.lower()
    sign = 1.0 if position == "long" else -1.0
    return sign * qty * (ST - F0)

# ---------- Break-even finder ----------
def find_breakevens(x, y, eps=1e-12):
    y2 = y.copy()
    y2[np.abs(y2) < eps] = 0.0
    s = np.sign(y2)

    last = 0.0
    for i in range(len(s)):
        if s[i] == 0.0:
            s[i] = last
        else:
            last = s[i]

    bes = []
    for i in range(len(s) - 1):
        if s[i] == 0.0 or s[i + 1] == 0.0:
            continue
        if s[i] != s[i + 1]:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            if y1 != y0:
                bes.append(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))

    bes = sorted(bes)
    cleaned = []
    for b in bes:
        if not cleaned or abs(b - cleaned[-1]) > 1e-6:
            cleaned.append(b)
    return cleaned

# ---------- Summary metrics ----------
def summarize_pnl(ST, pnl):
    min_pnl = float(np.min(pnl))
    max_pnl = float(np.max(pnl))
    idx_min = int(np.argmin(pnl))
    idx_max = int(np.argmax(pnl))
    return {
        "min_pnl": min_pnl,
        "max_pnl": max_pnl,
        "S_at_min": float(ST[idx_min]),
        "S_at_max": float(ST[idx_max]),
    }

def generate_conclusion(breakevens, stats):
    # Simple but useful conclusion
    lines = []
    if breakevens:
        lines.append(f"Break-even level(s): {', '.join([f'{b:.4f}' for b in breakevens])}.")
    else:
        lines.append("No break-even crossing found on the displayed range (PnL may be always positive or always negative).")

    lines.append(f"Max PnL on range: {stats['max_pnl']:.6f} at S_T≈{stats['S_at_max']:.4f}.")
    lines.append(f"Min PnL on range: {stats['min_pnl']:.6f} at S_T≈{stats['S_at_min']:.4f}.")

    if stats["max_pnl"] <= 0:
        lines.append("On this displayed range, the strategy is not profitable (PnL ≤ 0). Consider adjusting strikes, premiums, or direction.")
    elif stats["min_pnl"] >= 0:
        lines.append("On this displayed range, the strategy stays profitable (PnL ≥ 0).")
    else:
        lines.append("The strategy has both profit and loss regions depending on the final underlying price.")

    return " ".join(lines)

# ---------- Main run ----------
def run_once():
    print("\n=== STRATEGY P&L ENGINE (Options + Futures) ===")
    print("You define legs manually (no templates).")
    print("Outputs: total PnL at maturity, break-evens, and a clear conclusion.\n")

    # Market inputs (only needed for forward theoretical price if you use futures)
    S0 = ask_float("Spot price S0 (underlying price now): ")
    T  = ask_float("Time to maturity T (years, e.g. 0.5): ")
    r  = ask_percent("Risk-free interest rate r % (annualized) [0]: ", allow_negative=True, default="0")
    q  = ask_percent("Dividend yield / carry q % (annualized) [0]: ", allow_negative=True, default="0")

    if T <= 0:
        print("❌ T must be > 0.")
        return

    F0 = forward_price(S0, r, q, T)
    print(f"\nForward/Futures theoretical price F0 = S0 * exp((r - q) * T) = {F0:.6f}\n")

    # Legs definition
    n_legs = ask_int("How many legs? (integer >= 1): ", min_value=1)

    legs = []
    refs = [S0, F0]

    print("\nInstrument types:")
    print("  1 = Option (Call or Put)")
    print("  2 = Futures/Forward")

    for i in range(n_legs):
        print(f"\n--- Leg {i+1}/{n_legs} ---")
        inst = ask_int("Instrument type (1=Option, 2=Futures): ", valid=[1, 2])

        pos = ask_int("Position (1=Long, 2=Short): ", valid=[1, 2])
        position = "long" if pos == 1 else "short"

        qty = ask_float("Quantity (e.g. 1): ")
        if qty <= 0:
            print("❌ Quantity must be > 0.")
            return

        if inst == 1:
            # Option
            ot = ask_int("Option type (1=Call, 2=Put): ", valid=[1, 2])
            option_type = "call" if ot == 1 else "put"

            K = ask_float("Strike price K (exercise price): ")
            if K <= 0:
                print("❌ Strike price must be > 0.")
                return

            premium = ask_float("Option premium (price paid/received per option): ", allow_negative=False)

            legs.append({
                "kind": "option",
                "position": position,
                "qty": qty,
                "option_type": option_type,
                "K": K,
                "premium": premium
            })
            refs.append(K)

        else:
            # Futures
            print(f"Using theoretical F0={F0:.6f} (from S0, r, q, T).")
            legs.append({
                "kind": "futures",
                "position": position,
                "qty": qty,
                "F0": F0
            })
            refs.append(F0)

    # Build ST grid automatically (min input effort)
    min_ref = min(refs)
    max_ref = max(refs)
    spread = max(max_ref - min_ref, 0.5 * max_ref, 10.0)

    ST_min = 0.0
    ST_max = max_ref + spread
    n_points = int(np.clip((ST_max - ST_min) * 10, 2000, 25000))
    ST = np.linspace(ST_min, ST_max, n_points)

    # Compute total PnL
    pnl_total = np.zeros_like(ST, dtype=float)
    for leg in legs:
        if leg["kind"] == "option":
            pnl_total += pnl_option_at_maturity(
                ST,
                leg["option_type"],
                leg["K"],
                leg["premium"],
                leg["position"],
                leg["qty"]
            )
        else:
            pnl_total += pnl_futures_at_maturity(
                ST, leg["F0"], leg["position"], leg["qty"]
            )

    # Break-evens and stats
    bes = find_breakevens(ST, pnl_total)
    stats = summarize_pnl(ST, pnl_total)
    conclusion = generate_conclusion(bes, stats)

    # Print recap
    print("\n====================")
    print("Legs recap")
    print("====================")
    for idx, leg in enumerate(legs, 1):
        if leg["kind"] == "option":
            print(f"{idx}. OPTION {leg['option_type'].upper()} | {leg['position'].upper()} | qty={leg['qty']} | K={leg['K']} | premium={leg['premium']}")
        else:
            print(f"{idx}. FUTURES | {leg['position'].upper()} | qty={leg['qty']} | F0={leg['F0']:.6f}")

    print("\n====================")
    print("Results")
    print("====================")
    if bes:
        print("Break-even(s):", ", ".join([f"{b:.4f}" for b in bes]))
    else:
        print("Break-even(s): none detected on displayed range.")

    print(f"Max PnL (range): {stats['max_pnl']:.6f} at S_T≈{stats['S_at_max']:.4f}")
    print(f"Min PnL (range): {stats['min_pnl']:.6f} at S_T≈{stats['S_at_min']:.4f}")

    print("\nConclusion:")
    print(conclusion)

    # Plot
    plt.figure(figsize=(11, 6))
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    # green/red segments
    green = np.where(pnl_total >= 0, pnl_total, np.nan)
    red = np.where(pnl_total < 0, pnl_total, np.nan)
    plt.plot(ST, green, linewidth=2, label="PnL >= 0")
    plt.plot(ST, red, linewidth=2, label="PnL < 0")

    # break-even vertical lines
    for b in bes[:12]:
        plt.axvline(b, color="grey", linestyle=":", linewidth=1)

    plt.title("Strategy total P&L at maturity (Options + Futures)")
    plt.xlabel("Underlying price at maturity (S_T)")
    plt.ylabel("Profit / Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_name = "strategy_pnl.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n✅ Saved chart: {out_name}")

def main():
    while True:
        run_once()
        again = input("\nRun again? (y/n) [n]: ").strip().lower() or "n"
        if not again.startswith("y"):
            print("👋 Done.")
            break

if __name__ == "__main__":
    main()
