# strategy_pnl_engine.py
import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STRATEGY P&L ENGINE (Futures + Multi-leg Options)
# ------------------------------------------------------------
# What it does (distinct from the pricer):
# - Build any multi-leg strategy (options + futures)
# - Compute:
#   1) P&L at maturity: payoff - premium (options) + futures P&L
#   2) Mark-to-market P&L at time t < T using Black–Scholes valuation
# - Break-evens detection (automatic)
# - Aggregated Greeks (BS, at initial point)
# - Stress tests (Spot / Volatility / Time)
# - Optional P&L Heatmap (Spot x Time remaining)
# - Auto “conclusions” (risk profile, max loss/profit, key regions)
#
# Glossary (expanded):
# S0 : Spot price (underlying now)
# K  : Strike price
# r  : Risk-free interest rate (annualized, continuous)
# q  : Dividend yield / carry (annualized, continuous)
# sigma : Volatility (annualized)
# T  : Time to maturity (years)
# ============================================================


# -------------------------
# Robust inputs
# -------------------------
def ask_int(prompt, valid=None, min_value=None, max_value=None, default=None):
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return int(default)
        try:
            v = int(s)
            if min_value is not None and v < min_value:
                raise ValueError
            if max_value is not None and v > max_value:
                raise ValueError
            if valid is None or v in valid:
                return v
        except ValueError:
            pass
        msg = "❌ Invalid input."
        if valid is not None:
            msg += f" Valid choices: {sorted(valid)}"
        if min_value is not None:
            msg += f" | min={min_value}"
        if max_value is not None:
            msg += f" | max={max_value}"
        print(msg)

def ask_float(prompt, allow_negative=False, default=None):
    while True:
        s = input(prompt).strip().replace(",", ".")
        if s == "" and default is not None:
            return float(default)
        try:
            v = float(s)
            if allow_negative or v >= 0:
                return v
        except ValueError:
            pass
        print("❌ Enter a valid number.")

def ask_percent(prompt, allow_negative=False, default=None):
    v = ask_float(prompt, allow_negative=allow_negative, default=default)
    return v / 100.0


# -------------------------
# Normal CDF/PDF (no scipy)
# -------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# -------------------------
# Black–Scholes (European, continuous q)
# -------------------------
def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str) -> float:
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    if sigma <= 0:
        fwd = S * math.exp((r - q) * T)
        disc = math.exp(-r * T)
        return disc * (max(fwd - K, 0.0) if option_type == "call" else max(K - fwd, 0.0))

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        return S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    else:
        return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)

def bs_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str):
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    if T <= 0 or sigma <= 0:
        return {"Delta": 0.0, "Gamma": 0.0, "Vega_per_1pct": 0.0, "Theta_per_day": 0.0, "Rho_per_bp": 0.0}

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = norm_pdf(d1)

    if option_type == "call":
        delta = disc_q * norm_cdf(d1)
        rho = T * disc_r * K * norm_cdf(d2)
        theta = (-disc_q * S * pdf_d1 * sigma / (2.0 * sqrtT)
                 - r * disc_r * K * norm_cdf(d2)
                 + q * disc_q * S * norm_cdf(d1))
    else:
        delta = disc_q * (norm_cdf(d1) - 1.0)
        rho = -T * disc_r * K * norm_cdf(-d2)
        theta = (-disc_q * S * pdf_d1 * sigma / (2.0 * sqrtT)
                 + r * disc_r * K * norm_cdf(-d2)
                 - q * disc_q * S * norm_cdf(-d1))

    gamma = (disc_q * pdf_d1) / (S * sigma * sqrtT)
    vega = disc_q * S * pdf_d1 * sqrtT

    return {
        "Delta": float(delta),
        "Gamma": float(gamma),
        "Vega_per_1pct": float(vega / 100.0),
        "Theta_per_day": float(theta / 365.0),
        "Rho_per_bp": float(rho / 10000.0),
    }


# -------------------------
# Strategy legs and valuations
# -------------------------
def payoff_option(ST: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)

def pnl_leg_at_maturity(leg: dict, ST: np.ndarray) -> np.ndarray:
    sign = 1.0 if leg["position"] == "long" else -1.0
    qty = leg["qty"]

    if leg["kind"] == "futures":
        # Futures P&L = sign * qty * (ST - entry_price)
        return qty * sign * (ST - leg["F_entry"])

    if leg["kind"] == "option":
        payoff = payoff_option(ST, leg["K"], leg["option_type"])
        # P&L = sign * qty * (payoff - premium)
        return qty * sign * (payoff - leg["premium"])

    raise ValueError("Unknown leg kind.")

def bs_value_leg_now(leg: dict, S: float, r: float, q: float, sigma: float, T_rem: float) -> float:
    sign = 1.0 if leg["position"] == "long" else -1.0
    qty = leg["qty"]

    if leg["kind"] == "futures":
        # Mark-to-market value of futures position ≈ sign*qty*(S - entry_price)
        return qty * sign * (S - leg["F_entry"])

    if leg["kind"] == "option":
        v = bs_price(S, leg["K"], r, q, sigma, T_rem, leg["option_type"])
        return qty * sign * v

    raise ValueError("Unknown leg kind.")

def bs_value_strategy(legs, S: float, r: float, q: float, sigma: float, T_rem: float) -> float:
    return sum(bs_value_leg_now(leg, S, r, q, sigma, T_rem) for leg in legs)

def greeks_strategy_init(legs, S0, r, q, sigma, T):
    totals = {"Delta": 0.0, "Gamma": 0.0, "Vega_per_1pct": 0.0, "Theta_per_day": 0.0, "Rho_per_bp": 0.0}
    for leg in legs:
        sign = 1.0 if leg["position"] == "long" else -1.0
        qty = leg["qty"]
        if leg["kind"] == "option":
            g = bs_greeks(S0, leg["K"], r, q, sigma, T, leg["option_type"])
            for k in totals:
                totals[k] += qty * sign * g[k]
        elif leg["kind"] == "futures":
            # Futures: Delta ≈ sign*qty, others ~0
            totals["Delta"] += qty * sign * 1.0
    return totals


# -------------------------
# Break-evens (crossings)
# -------------------------
def find_breakevens_crossings(x: np.ndarray, y: np.ndarray, eps=1e-12):
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
                bes.append(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))

    bes = sorted(bes)
    cleaned = []
    for b in bes:
        if not cleaned or abs(b - cleaned[-1]) > 1e-6:
            cleaned.append(b)
    return cleaned


# -------------------------
# Conclusions engine
# -------------------------
def summarize_pnl_curve(x, y, label=""):
    max_p = float(np.nanmax(y))
    min_p = float(np.nanmin(y))
    idx_max = int(np.nanargmax(y))
    idx_min = int(np.nanargmin(y))
    x_max = float(x[idx_max])
    x_min = float(x[idx_min])

    # Identify profit region fraction
    frac_pos = float(np.mean(y >= 0.0))
    return {
        "label": label,
        "max_pnl": max_p, "max_at": x_max,
        "min_pnl": min_p, "min_at": x_min,
        "frac_positive": frac_pos,
    }

def print_conclusions(summary, breakevens, x_name):
    print("\n====================")
    print("Conclusions (auto)")
    print("====================")
    print(f"- Max P&L : {summary['max_pnl']:.4f} (at {x_name} ≈ {summary['max_at']:.4f})")
    print(f"- Min P&L : {summary['min_pnl']:.4f} (at {x_name} ≈ {summary['min_at']:.4f})")

    if math.isfinite(summary["max_pnl"]) and math.isfinite(summary["min_pnl"]):
        if abs(summary["max_pnl"]) > 1e12 or abs(summary["min_pnl"]) > 1e12:
            print("- ⚠️ P&L range is huge: check quantities and units.")

    if breakevens:
        be_str = ", ".join([f"{b:.4f}" for b in breakevens[:8]])
        print(f"- Break-even(s): {be_str}")
        if len(breakevens) > 8:
            print(f"  (and {len(breakevens)-8} more...)")
    else:
        print("- Break-even(s): none found on the plotted range.")

    if summary["frac_positive"] > 0.7:
        print("- Profit region dominates (>=70% of plotted range).")
    elif summary["frac_positive"] < 0.3:
        print("- Loss region dominates (>=70% of plotted range).")
    else:
        print("- Mixed profile: both profit/loss zones are significant.")

    if summary["min_pnl"] < 0 and summary["max_pnl"] > 0:
        print("- Strategy has both upside and downside (non-trivial risk profile).")
    elif summary["max_pnl"] <= 0:
        print("- Strategy appears structurally losing on the plotted range (verify premiums / direction).")
    elif summary["min_pnl"] >= 0:
        print("- Strategy appears always positive on the plotted range (verify range + assumptions).")


# -------------------------
# Stress tests (MTM)
# -------------------------
def run_stress_tests(legs, S0, r, q, sigma, T, T_rem):
    base_init = bs_value_strategy(legs, S0, r, q, sigma, T)      # value at inception (BS)
    base_now  = bs_value_strategy(legs, S0, r, q, sigma, T_rem)  # value now (BS)
    base_pnl  = base_now - base_init

    print("\n====================")
    print("Stress tests (Mark-to-market, BS)")
    print("====================")
    print(f"Base: Value_init={base_init:.6f} | Value_now={base_now:.6f} | MTM P&L={base_pnl:.6f}")
    print(f"{'Type':<10} | {'Shock':<14} | {'MTM P&L vs init':>18}")
    print("-"*50)

    spot_shocks = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    vol_shocks  = [-0.10, -0.05, 0.0, 0.05, 0.10]  # absolute (e.g. +0.05 = +5 vol points)
    time_shocks = [0.0, -T/12, -T/6, -T/4]

    for ds in spot_shocks:
        S = S0 * (1.0 + ds)
        v_now = bs_value_strategy(legs, S, r, q, sigma, T_rem)
        pnl = v_now - base_init
        print(f"{'Spot':<10} | {ds:+.0%:<14} | {pnl:>18.6f}")

    for dv in vol_shocks:
        sig = max(1e-8, sigma + dv)
        v_now = bs_value_strategy(legs, S0, r, q, sig, T_rem)
        pnl = v_now - base_init
        print(f"{'Volatility':<10} | {dv:+.0%}pt{'':<8} | {pnl:>18.6f}")

    for dt in time_shocks:
        Trem = max(1e-8, T_rem + dt)
        v_now = bs_value_strategy(legs, S0, r, q, sigma, Trem)
        pnl = v_now - base_init
        print(f"{'Time':<10} | {dt:+.4f}y{'':<6} | {pnl:>18.6f}")


# -------------------------
# Heatmap (Spot x Time remaining)
# -------------------------
def plot_pnl_heatmap(legs, S0, r, q, sigma, T):
    S_grid = np.linspace(max(1e-9, 0.5*S0), 1.5*S0, 80)
    T_grid = np.linspace(1e-6, T, 60)

    base_init = bs_value_strategy(legs, S0, r, q, sigma, T)

    Z = np.zeros((len(T_grid), len(S_grid)))
    for i, Trem in enumerate(T_grid):
        for j, S in enumerate(S_grid):
            Z[i, j] = bs_value_strategy(legs, S, r, q, sigma, Trem) - base_init

    plt.figure(figsize=(10, 6))
    plt.imshow(Z, aspect="auto", origin="lower",
               extent=[S_grid[0], S_grid[-1], T_grid[0], T_grid[-1]])
    plt.colorbar(label="MTM P&L = Value_now - Value_init (BS)")
    plt.title("P&L Heatmap (Spot x Time remaining)")
    plt.xlabel("Spot price (S)")
    plt.ylabel("Time remaining (years)")
    plt.tight_layout()
    plt.show()


# -------------------------
# Main run
# -------------------------
def run_once():
    print("\n=== STRATEGY P&L ENGINE (Futures + Multi-leg Options) ===")

    # Market inputs
    print("\nMarket parameters (percent input: type 3 for 3%)")
    S0 = ask_float("Spot price S0 (underlying now): ")
    r = ask_percent("Risk-free interest rate r % (annualized) [0]: ", allow_negative=True, default=0)
    q = ask_percent("Dividend yield / carry q % (annualized) [0]: ", allow_negative=True, default=0)
    sigma = ask_percent("Volatility sigma % (annualized): ", default=None)
    T = ask_float("Time to maturity T (years, e.g. 0.5): ")

    if T <= 0:
        print("❌ T must be > 0.")
        return

    # MTM date selection
    print("\nMark-to-market date (t < T):")
    mode = ask_int("1 = input elapsed time t | 2 = input remaining time T_rem  [2]: ", valid=[1,2], default=2)
    if mode == 1:
        t = ask_float("Elapsed time t (years, e.g. 0.1): ")
        T_rem = T - t
    else:
        T_rem = ask_float("Remaining time T_rem (years, e.g. 0.3): ")

    if T_rem <= 0 or T_rem > T:
        print("❌ Remaining time must satisfy 0 < T_rem <= T.")
        return

    # Strategy legs
    print("\nBuild strategy legs:")
    print("  Instrument types:")
    print("   1 = Futures (linear)")
    print("   2 = Call option (premium required)")
    print("   3 = Put option  (premium required)")
    n_legs = ask_int("Number of legs: ", min_value=1)

    legs = []
    ref_prices = [S0]

    for i in range(n_legs):
        print(f"\n--- Leg {i+1}/{n_legs} ---")
        pos = ask_int("Position: 1=Short | 2=Long  [2]: ", valid=[1,2], default=2)
        position = "short" if pos == 1 else "long"
        qty = ask_float("Quantity (contracts/units, e.g. 1): ")
        if qty <= 0:
            print("❌ Quantity must be > 0.")
            return

        inst = ask_int("Instrument: 1=Futures | 2=Call | 3=Put : ", valid=[1,2,3])

        if inst == 1:
            # Entry price default = S0 (or user can override)
            F_entry = ask_float("Futures entry price (default = S0) [S0]: ", default=S0)
            legs.append({"kind":"futures", "position":position, "qty":qty, "F_entry":F_entry})
            ref_prices.append(F_entry)

        else:
            option_type = "call" if inst == 2 else "put"
            K = ask_float("Strike price K: ")
            if K <= 0:
                print("❌ Strike must be > 0.")
                return

            # Premium: auto computed using BS unless user overrides (minimal input)
            prem_bs = bs_price(S0, K, r, q, sigma, T, option_type)
            use_auto = ask_int(f"Use automatic premium (Black–Scholes) = {prem_bs:.6f}? 1=Yes | 0=No  [1]: ", valid=[0,1], default=1)
            if use_auto == 1:
                premium = prem_bs
            else:
                premium = ask_float("Premium paid/received (absolute): ", allow_negative=False)

            legs.append({
                "kind":"option", "position":position, "qty":qty,
                "option_type":option_type, "K":K, "premium":premium
            })
            ref_prices.append(K)

    # ==========
    # P&L at maturity (payoff - premium)
    # ==========
    span = max(max(ref_prices) * 0.5, 10.0)
    ST_min = max(0.0, min(ref_prices) - span)
    ST_max = max(ref_prices) + span
    n_points = int(np.clip((ST_max - ST_min) * 10, 1500, 25000))
    ST = np.linspace(ST_min, ST_max, n_points)

    pnl_T = np.zeros_like(ST)
    for leg in legs:
        pnl_T += pnl_leg_at_maturity(leg, ST)

    be_T = find_breakevens_crossings(ST, pnl_T)
    sum_T = summarize_pnl_curve(ST, pnl_T, label="Maturity P&L")
    print_conclusions(sum_T, be_T, x_name="S_T")

    # ==========
    # MTM P&L curve at time t (BS valuation)
    # ==========
    value_init = bs_value_strategy(legs, S0, r, q, sigma, T)
    S_curve = np.linspace(max(1e-9, 0.5*S0), 1.5*S0, 600)
    pnl_mtm = np.array([bs_value_strategy(legs, s, r, q, sigma, T_rem) - value_init for s in S_curve])
    be_mtm = find_breakevens_crossings(S_curve, pnl_mtm)
    sum_mtm = summarize_pnl_curve(S_curve, pnl_mtm, label="MTM P&L")

    print("\n====================")
    print("Mark-to-market (BS)")
    print("====================")
    print(f"T (initial maturity) = {T:.6f} | Remaining time T_rem = {T_rem:.6f}")
    v_now_S0 = bs_value_strategy(legs, S0, r, q, sigma, T_rem)
    print(f"Value init (t=0, S0) : {value_init:.6f}")
    print(f"Value now  (t, S0)   : {v_now_S0:.6f}")
    print(f"MTM P&L   (t, S0)    : {(v_now_S0 - value_init):.6f}")
    print_conclusions(sum_mtm, be_mtm, x_name="S_t")

    # ==========
    # Greeks
    # ==========
    g = greeks_strategy_init(legs, S0, r, q, sigma, T)
    print("\n====================")
    print("Aggregated Greeks (Black–Scholes, at inception)")
    print("====================")
    print(f"Delta          : {g['Delta']:.6f}")
    print(f"Gamma          : {g['Gamma']:.6f}")
    print(f"Vega (+1% vol) : {g['Vega_per_1pct']:.6f}")
    print(f"Theta (per day): {g['Theta_per_day']:.6f}")
    print(f"Rho (per bp)   : {g['Rho_per_bp']:.8f}")

    # ==========
    # Stress tests
    # ==========
    do_stress = ask_int("\nRun stress tests (Spot/Volatility/Time)? 1=Yes | 0=No  [1]: ", valid=[0,1], default=1)
    if do_stress == 1:
        run_stress_tests(legs, S0, r, q, sigma, T, T_rem)

    # ==========
    # Plots
    # ==========
    print("\nPlots:")
    print("  1 = P&L at maturity (vs S_T)")
    print("  2 = MTM P&L (vs S_t)")
    print("  3 = Both")
    print("  4 = Heatmap (Spot x Time remaining)")
    print("  5 = All")
    print("  0 = None")
    p = ask_int("Choice [5]: ", valid=[0,1,2,3,4,5], default=5)

    if p in (1,3,5):
        plt.figure(figsize=(11, 6))
        plt.axhline(0, color="black", ls="--", lw=1)
        green = np.where(pnl_T >= 0, pnl_T, np.nan)
        red   = np.where(pnl_T < 0, pnl_T, np.nan)
        plt.plot(ST, green, lw=2, label="P&L at maturity >= 0")
        plt.plot(ST, red,   lw=2, label="P&L at maturity < 0")
        for b in be_T[:8]:
            plt.axvline(b, color="grey", ls=":", lw=1)
        plt.title("Strategy P&L at maturity (payoff - premium)")
        plt.xlabel("Underlying price at maturity (S_T)")
        plt.ylabel("Profit / Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("strategy_pnl_maturity.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("✅ Saved: strategy_pnl_maturity.png")

    if p in (2,3,5):
        plt.figure(figsize=(11, 6))
        plt.axhline(0, color="black", ls="--", lw=1)
        green = np.where(pnl_mtm >= 0, pnl_mtm, np.nan)
        red   = np.where(pnl_mtm < 0, pnl_mtm, np.nan)
        plt.plot(S_curve, green, lw=2, label="MTM P&L >= 0 (BS)")
        plt.plot(S_curve, red,   lw=2, label="MTM P&L < 0 (BS)")
        for b in be_mtm[:8]:
            plt.axvline(b, color="grey", ls=":", lw=1)
        plt.title(f"Mark-to-market P&L (Black–Scholes) — Remaining time T_rem={T_rem:.4f}y")
        plt.xlabel("Spot price now (S_t)")
        plt.ylabel("MTM P&L = Value_now - Value_init")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("strategy_pnl_mtm.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("✅ Saved: strategy_pnl_mtm.png")

    if p in (4,5):
        plot_pnl_heatmap(legs, S0, r, q, sigma, T)

def main():
    while True:
        run_once()
        again = input("\nRun again? (y/n) [n]: ").strip().lower() or "n"
        if not again.startswith("y"):
            print("👋 Done.")
            break

if __name__ == "__main__":
    main()