# Options Pricing & Strategy P&L Engine (Python)

This repository contains two independent Python tools designed for option pricing and multi-leg strategy analysis, using industry-standard quantitative methods.

The objective is to provide clean, transparent and educational implementations of:
- Black–Scholes pricing
- Monte Carlo simulation
- Options and futures payoff analysis

The focus is on financial reasoning, risk understanding, and clear visualisation, rather than black-box libraries.

---------------------------------------------------------------------

REPOSITORY STRUCTURE

.
├── option_pricer.py
├── strategy_pnl_engine.py
└── README.md

Each script is fully standalone and can be executed independently.

---------------------------------------------------------------------

1) OPTION PRICER – BLACK–SCHOLES & MONTE CARLO

PURPOSE

This script prices European options using:
- Black–Scholes closed-form formula
- Monte Carlo simulation under the risk-neutral measure

It also provides intuitive graphical visualisations of pricing uncertainty.

---------------------------------------------------------------------

MODEL INPUTS

The user defines the following market parameters:

- Spot price (S0): current price of the underlying asset
- Strike price (K): option strike
- Volatility (sigma): annualized volatility
- Risk-free rate (r): continuously compounded interest rate
- Dividend yield (q): continuous dividend or carry
- Time to maturity (T): expressed in years
- Number of simulations: number of Monte Carlo paths

---------------------------------------------------------------------

METHODOLOGY

- Black–Scholes pricing with continuous dividend yield
- Monte Carlo simulation of the terminal price ST
- Discounted expected payoff under risk-neutral dynamics
- 95% confidence interval for the Monte Carlo estimator

---------------------------------------------------------------------

VISUAL OUTPUT

- Scatter plot of all Monte Carlo simulated option prices
- Thick horizontal line representing the Monte Carlo mean price
- Black–Scholes price displayed for direct comparison
- Clear visual intuition of dispersion and convergence

---------------------------------------------------------------------

WHAT THIS DEMONSTRATES

- Solid understanding of option pricing theory
- Ability to implement quantitative models from scratch
- Quantitative reasoning and model validation skills
- Clear communication through visualisation

---------------------------------------------------------------------

2) STRATEGY P&L ENGINE – MULTI-LEG OPTIONS & FUTURES

PURPOSE

This script analyses the profit and loss profile of custom multi-leg strategies composed of:
- European calls
- European puts
- Futures or forwards

Strategies are built leg by leg, without relying on predefined templates.

---------------------------------------------------------------------

STRATEGY CONSTRUCTION

For each leg, the user specifies:
- Instrument type: Call, Put or Future
- Position: Long or Short
- Quantity
- Strike price (for options)
- Premium paid or received

There is no limit on the number of legs.

---------------------------------------------------------------------

METHODOLOGY

- Payoff computation at maturity for each leg
- Aggregation into a total strategy P&L profile
- Automatic detection of break-even points
- Identification of profit and loss regions

---------------------------------------------------------------------

VISUAL OUTPUT

- Strategy P&L curve at maturity
- Green areas represent profit zones
- Red areas represent loss zones
- Vertical lines highlight break-even levels

---------------------------------------------------------------------

WHAT THIS DEMONSTRATES

- Understanding of option payoff structures and convexity
- Risk decomposition of structured products
- Strategy construction logic and intuition
- Ability to analyse downside and upside asymmetry

---------------------------------------------------------------------

HOW TO RUN

Make sure you have Python 3.9 or higher installed.

Install required dependencies:

pip install numpy matplotlib

Run each script independently:

python option_pricer.py
python strategy_pnl_engine.py

---------------------------------------------------------------------

AUTHOR NOTES

- No external pricing libraries are used (no SciPy, QuantLib, etc.)
- All models are implemented explicitly for transparency
- Focus on clarity, financial logic and robustness
- Scripts are designed for educational and analytical purposes

---------------------------------------------------------------------

POSSIBLE EXTENSIONS

- Greeks aggregation for multi-leg strategies
- Implied volatility solver
- Mark-to-market P&L before maturity
- Stress testing on spot, volatility and time
- Streamlit or dashboard implementation

---------------------------------------------------------------------

DISCLAIMER

This project is for educational and analytical purposes only.
It does not constitute financial advice or a trading recommendation.
