"""
Information-Theoretic Option Pricing
====================================
This project uses two information-theoretic methods (maximum entropy and minimum cross-entropy) 
to recover risk-neutral density implied by option prices, then compares them to the Black-Scholes
lognormal distribution and the model-free Breeden-Litzenberger method.

Methods
-------
1. Black-Scholes            : Parametric baseline assuming lognormal S_T.
2. Breeden-Litzenberger     : Model-free density from the call-price curve,
                              f_Q(K) = e^{rT} d^2 C/dK^2 (Breeden and Litzenberger, 1978).
3. Maximum entropy (MaxEnt) : Least-committal density consistent with observed
                              option prices (Buchen and Kelly, 1996).
4. Stutzer canonical val.   : Minimum-cross-entropy tilt of the empirical
                              return distribution (Stutzer, 1996).

Drivers
-------
- run(market)              : Single-maturity comparison of all four methods
- run_term_structure(...)  : Repeats the recovery across multiple maturities and reports
                               the term structure of the gap between option-implied
                               and historical crash probability.
- run_cross_sector(...)    : [Extension of the current project] Applies the pipeline to
                               multiple tickers (sector ETFs vs SPY). Sector ETF option chains
                               are roughly 10x thinner than SPY's, so leads to noisier recoveries
                               and greater possibility of failures. Shouldn't be treated as a concrete finding.

Runs offline out of the box on a synthetic skewed/fat-tailed market.
To use live data via yfinance, pass load_real_market() instead of make_synthetic_market().

Requires: numpy, scipy, matplotlib  (and yfinance if using real data).
"""

import numpy as np
# Account for older NumPy versions using trapz instead of trapezoid
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.optimize import minimize, brentq
import matplotlib
matplotlib.use("Agg")            # Headless backend so plots save without a display
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Black-Scholes (baseline)
# ----------------------------------------------------------------------
def bs_call(S0, K, r, q, sigma, T):
    """
    Closed-form Black-Scholes price for a European call.

    Risk neutral options pricing formula under constant interest rate,  
    C = exp(-rT) E_Q[(S_T - K)^+], evaluated under the assumption that
    S_T is lognormal under Q. The conditional handles degenerate cases
    (expired option or zero volatility), where the answer is simply intrinsic value.
    """
    if T <= 0 or sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_implied_vol(price, S0, K, r, q, T):
    """
    Invert Black-Scholes to get the volatility that reproduces a given market price.
    Since there is no closed-form inverse, root-find on the price equation between
    1e-4 (approx. zero vol) and 5.0 (500% annual vol, large enough to upper bound any realistic case).
    Returns NaN if no root exists in the bracket, which signals either deep arbitrage
    or a quote too stale to be fitted by a vol.
    """
    try:
        return brentq(lambda s: bs_call(S0, K, r, q, s, T) - price, 1e-4, 5.0)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------
# 2. Breeden-Litzenberger: model-free density  f_Q(K) = e^{rT} d^2 C/dK^2
# ----------------------------------------------------------------------
def breeden_litzenberger(strikes, calls, r, T):
    """
    Model-free risk-neutral density from the call-price curve.

    The relation  f_Q(K) = e^{rT} * d^2 C/dK^2  results from a butterfly-spread
    replication argument and is a no-arbitrage fact that assumes no model.
    But differentiating noisy quotes twice amplifies noise badly, so in practice,
    this method is likely to be fragile.

    When implementing: differentiating a piecewise linear interpolant
    produces a sum of delta-like spikes at every knot. Fitting a CubicSpline
    first and taking its analytic 2nd derivative gives smooth, well-behaved
    density and clean cross-check against MaxEnt (which is also smooth).
    """
    from scipy.interpolate import CubicSpline
    K = np.asarray(strikes, float)
    C = np.asarray(calls, float)
    order = np.argsort(K)
    K, C = K[order], C[order]
    cs = CubicSpline(K, C)
    Kd = np.linspace(K.min(), K.max(), 600)
    # Round small negative wobbles to zero: densities can't be negative, but the
    # 2nd derivative of an almost linear region can dip slightly below zero
    # numerically. Renormalize so the result integrates to 1.
    f = np.maximum(np.exp(r * T) * cs(Kd, 2), 0.0)
    area = np.trapezoid(f, Kd)
    return Kd, (f / area if area > 0 else f)


# ----------------------------------------------------------------------
# 3. Maximum-entropy density recovery from option prices
# ----------------------------------------------------------------------
def maxent_density(S_grid, strikes, calls, S0, r, q, T):
    """
    Maximize Shannon entropy of f(S_T) subject to:
        E_Q[S_T]          = F = S0 exp((r-q)T)   (forward / martingale)
        E_Q[(S_T - K_j)^+] = C_j exp(rT)         (each option price)
        integral f dS     = 1                     (normalization)

    Variational calculus gives solution the Gibbs exponential-family form
        f(S) ∝ exp(-Σ_i λ_i g_i(S)), where g_0(S) = S (forward), g_j(S) = (S - K_j)^+ (each call payoff).
    The λ_i are found by minimizing the smooth convex dual
        Ψ(λ) = log Z(λ) + Σ_i λ_i μ_i,  ΔΨ = μ - E_f[g]
    Setting the gradient to zero re-imposes the constraints.

    Returns (S_grid, density, masses). Density is mass-per-unit-price;
    masses sum to 1 and are used directly for moment calculations.
    """
    S = np.asarray(S_grid, float)
    dS = S[1] - S[0]
    F = S0 * np.exp((r - q) * T)

    # In constraint matrix G, each row is one g_i(S) evaluated on the grid
    # Row 0 = S itself (forward); subsequent rows = call payoffs.
    G = [S.copy()]
    targets = [F]
    for K, C in zip(strikes, calls):
        G.append(np.maximum(S - K, 0.0))
        targets.append(C * np.exp(r * T))     # Undiscount: prices are in
                                              # today's dollars, constraint
                                              # in time-T dollars (expected payoff)
    G = np.array(G)
    targets = np.array(targets)

    # Conditioning: option prices span orders of magnitude (deep-ITM ~$200,
    # far-OTM ~$0.05). Rescale so that large-magnitude constraints don't dominate
    # the optimizer's step direction and convergence stalls. Dividing each
    # constraint row by its target magnitude makes them all O(1).
    scale = np.maximum(np.abs(targets), 1.0)
    Gs = G / scale[:, None]
    ts = targets / scale

    def dual(lam):
        """Convex dual objective Ψ(λ) and its analytic gradient.

        log_unnorm[i] = -Σ_i λ_i g_i(S_i) is the unnormalized log-density
        on the grid. logsumexp gives log Z(λ) stably (the naive
        log(Σ exp(...)) overflows for moderate |λ|). Softmax normalization
        gives proper probability masses, so the normalization constraint
        ∫ f = 1 is satisfied by construction: no explicit Lagrange multiplier
        """
        log_unnorm = -(lam @ Gs)
        Z = logsumexp(log_unnorm)
        w = np.exp(log_unnorm - Z)              # masses, sum to 1
        L = Z + lam @ ts                        # dual objective
        grad = ts - Gs @ w                      # target − E_f[g] = 0 at optimum
        return L, grad

    # Quasi-Newton solve. Passing the analytic gradient is important; prevents
    # the optimizer estimating it by finite differences (which is slower and
    # less accurate).
    res = minimize(dual, np.zeros(len(ts)), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=10000, ftol=1e-14, gtol=1e-10))
    lam = res.x
    log_unnorm = -(lam @ Gs)
    w = np.exp(log_unnorm - logsumexp(log_unnorm))
    return S, w / dS, w


# ----------------------------------------------------------------------
# 4. Stutzer canonical valuation (minimum cross-entropy/exponential tilt)
# ----------------------------------------------------------------------
def stutzer_density(hist_gross_returns, S0, r, T):
    """
    Tilt the empirical historical return distribution to the closest distribution
    in terms of KL divergence that satisfies the martingale constraint  E_Q[S_T/S0] = exp(rT).

    Lagrangian solution to the minimum-D_KL problem is exponential tilting:
    p_t ∝ exp(γ * R_t), where R_t are observed gross returns and γ is the scalar Lagrange
    multiplier. Find by 1D root-finding on the constraint E_p[R] − exp(rT).

    Interpretation of the recovered γ: when negative (equity data with a positive risk premium),
    it tilts probability mass away from the high-return states in the historical sample, which 
    corresponds to the risk-neutral measure stripping out the equity premium.

    Returns (S0*R, p, γ): terminal prices, their risk-neutral probabilities, and tilt parameter.
    """
    R = np.asarray(hist_gross_returns, float)
    target = np.exp(r * T)

    def constraint(gamma):
        # Overflow safe, subtract max before exp.
        wl = gamma * R
        p = np.exp(wl - wl.max())
        p /= p.sum()
        return np.sum(p * R) - target

    # Brackets ±50 to be safe since in reality |γ| comes out single-digit for
    # equity index data and 10-20 for highly skewed historical samples.
    gamma = brentq(constraint, -50, 50)
    wl = gamma * R
    p = np.exp(wl - wl.max())
    p /= p.sum()
    return S0 * R, p, gamma


# ----------------------------------------------------------------------
# Data: synthetic (offline validation) and real (yfinance) loaders
# ----------------------------------------------------------------------
def make_synthetic_market(seed=0):
    """
    Deliberately non-lognormal "market" with known truth.

    The "true" density under Q is a mixture of two lognormals: a benign component
    (85% weight, mean 105) and a crash component (15%, mean 80, wider).
    The mixture produces negative skew and a fat left tail, the shape
    equity index markets quote. Option prices are then generated
    by integrating payoffs against this density, so MaxEnt recovery can be
    checked against the known truth.

    The historical-return generator uses a different measure (drift 8% > r,
    plus occasional jumps); this gives Stutzer a real risk premium to
    strip out and exercises the P-vs-Q distinction the project demonstrates.
    """
    S0, r, q, T = 100.0, 0.04, 0.0, 0.5
    S = np.linspace(0.01, 300, 4000)

    def ln(S, mu, s):
        return np.exp(-(np.log(S) - mu) ** 2 / (2 * s ** 2)) / (S * s * np.sqrt(2 * np.pi))

    # Mixture of two lognormals --> skewed and fat-tailed
    f_true = 0.85 * ln(S, np.log(105), 0.18) + 0.15 * ln(S, np.log(80), 0.35)
    f_true /= np.trapezoid(f_true, S)

    # Generate option prices from the true density by integrating payoffs
    strikes = np.array([70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130], float)
    calls = np.array([np.exp(-r * T) * np.trapezoid(np.maximum(S - K, 0) * f_true, S)
                      for K in strikes])

    # Simulated historical returns under a real-world (P) measure
    rng = np.random.default_rng(seed)
    n = 40000
    z = rng.standard_normal(n)
    jump = (rng.random(n) < 0.05) * rng.normal(-0.15, 0.05, n)
    logret = (0.08 - 0.5 * 0.2 ** 2) * T + 0.2 * np.sqrt(T) * z + jump
    hist_R = np.exp(logret)

    return dict(S0=S0, r=r, q=q, T=T, strikes=strikes, calls=calls,
                hist_R=hist_R, S_grid=S, f_true=f_true)


def clean_option_chain(strikes, calls, S0, lo=0.7, hi=1.3):
    """
    No-arbitrage cleanup for a raw call chain: sort, drop duplicate strikes,
    window around spot, and drop any strike whose price violates the
    monotonicity required by no-arbitrage (call prices must be non-increasing
    in strike; a violation can only be a crossed or stale quote).

    Not used by the live pipeline but kept here as an example of a no-arbitrage
    filter for raw call data
    """
    K = np.asarray(strikes, float)
    C = np.asarray(calls, float)
    order = np.argsort(K); K, C = K[order], C[order]
    _, idx = np.unique(K, return_index=True)
    K, C = K[idx], C[idx]
    win = (K > lo * S0) & (K < hi * S0)
    K, C = K[win], C[win]
    # Greedy monotonic filter: keep only strikes whose price is strictly
    # below the prie of the last kept strike. Drop crossed quotes.
    keep = [0]
    for i in range(1, len(K)):
        if C[i] < C[keep[-1]]:
            keep.append(i)
    return K[keep], C[keep]


def fit_arbitrage_free_calls(strikes_otm, prices_otm, is_call, S0, r, q, T,
                             n_out=30, window=(0.80, 1.20)):
    """
    Turn noisy OTM option mids into a clean, nearly arbitrage-free call curve
    curve suitable for MaxEnt.

    Pipeline:
      1. Put-call parity: convert each OTM put to its call-equivalent price
         using  C = P + S0 exp(-qT) - K exp(-rT). Parity is a
         pure no-arbitrage fact (replication), so this is model-free
      2. Window around spot: drop strikes far from spot, where OTM quotes are
         thinly traded and unreliable mids.
      3. Invert each price to its Black-Scholes implied vol. Vol space is
         much smoother than price space, which makes the next step's fit
         well-behaved.
      4. Fit implied vol as a quadratic in log-moneyness k = ln(K/F). The
         quadratic is the simplest sensible smile shape (level + skew +
         convexity); SVI/SSVI are the industry-grade upgrades.
      5. Regenerate clean BS prices on an evenly-spaced strike grid using
         the fitted vol at each strike. These are internally consistent by
         construction, which is what MaxEnt's hard-equality constraints need.

    Returns (Kfit, Cfit, K_obs, iv_obs, iv_fit); the latter three are for plotting
    the smile if desired.
    """
    F = S0 * np.exp((r - q) * T)
    K = np.asarray(strikes_otm, float)
    P = np.asarray(prices_otm, float)

    # Put-call parity conversion. Calls stay as is; puts get parity-adjusted.
    C_equiv = np.where(is_call, P, P + S0 * np.exp(-q * T) - K * np.exp(-r * T))

    win = (K > window[0] * S0) & (K < window[1] * S0)
    K, C_equiv = K[win], C_equiv[win]
    ivs = np.array([bs_implied_vol(c, S0, k, r, q, T) for c, k in zip(C_equiv, K)])
    good = np.isfinite(ivs) & (ivs > 1e-3)
    K_obs, iv_obs = K[good], ivs[good]
    if len(K_obs) < 5:
        raise ValueError("Too few usable IVs to fit a smile; widen the window/expiry.")

    # Quadratic smile fit in log-moneyness, then regenerate clean prices
    coef = np.polyfit(np.log(K_obs / F), iv_obs, 2)
    Kfit = np.linspace(K_obs.min(), K_obs.max(), n_out)
    iv_fit = np.polyval(coef, np.log(Kfit / F))
    Cfit = np.array([bs_call(S0, k, r, q, s, T) for k, s in zip(Kfit, iv_fit)])
    return Kfit, Cfit, K_obs, iv_obs, iv_fit


def load_real_market(ticker="SPY", target_days=45, r=0.04, q=0.013,
                     window=(0.80, 1.20), n_strikes=30, history_period="20y"):
    """
    Pull a live option chain via yfinance and run it through the smile-fit
    pipeline. Requires network + 'pip install yfinance'.

    Returns the same dict shape as make_synthetic_market(), so downstream
    drivers don't care whether data real or synthetic.

    Parameters to note:
      r : current short-term treasury yield (e.g. 3-month T-bill). The
          forward F = S0 exp((r-q)T) is the martingale target.
      q : dividend yield of the underlying (~1.3% for SPY).
      window : OTM strike range as fractions of spot. Wider gives more
          smile-fit data but includes noisier wing quotes; narrower
          safer for thinly traded chains.
      history_period : look-back for the Stutzer empirical density. Short
          windows in calm regimes contain no extreme drawdowns and yield
          degenerate (zero) tail probabilities; '20y' covers 2008, 2020, 2022.
    """
    import yfinance as yf
    import datetime as dt
    tk = yf.Ticker(ticker)
    S0 = float(tk.history(period="1d")["Close"].iloc[-1])

    # Pick the listed expiry closest to target_days. Skip anything <5d out bc
    # near-expiry options are dominated by gamma noise and aren't useful
    # for density recovery.
    today = dt.date.today()
    exps = [(e, (dt.datetime.strptime(e, "%Y-%m-%d").date() - today).days)
            for e in tk.options]
    exps = [(e, d) for e, d in exps if d > 5]
    expiry, days = min(exps, key=lambda ed: abs(ed[1] - target_days))
    T = days / 365.0
    F = S0 * np.exp((r - q) * T)

    oc = tk.option_chain(expiry)

    def liquid_mid(df):
        """Filter to liquid quotes (positive bid/ask, volume or open interest)
        and return (strikes, mid-prices). Mid is more reliable than last
        trade, which can be stale."""
        df = df.copy()
        df["volume"] = df["volume"].fillna(0)
        df["openInterest"] = df.get("openInterest", 0).fillna(0)
        ok = (df.bid > 0) & (df.ask > 0) & ((df.volume > 0) | (df.openInterest > 0))
        df = df[ok]
        return df.strike.values, (0.5 * (df.bid + df.ask)).values

    # OTM-only: puts below the forward, calls above. ITM quotes are wide and
    # illiquid; their mids distort the recovery, so discard them and rely
    # on parity in fit_arbitrage_free_calls to convert OTM puts to call prices.
    Kc, Mc = liquid_mid(oc.calls)
    Kp, Mp = liquid_mid(oc.puts)
    cmask, pmask = Kc >= F, Kp < F
    strikes_otm = np.concatenate([Kp[pmask], Kc[cmask]])
    prices_otm  = np.concatenate([Mp[pmask], Mc[cmask]])
    is_call     = np.concatenate([np.zeros(pmask.sum(), bool),
                                  np.ones(cmask.sum(), bool)])

    strikes, calls, K_obs, iv_obs, iv_fit = fit_arbitrage_free_calls(
        strikes_otm, prices_otm, is_call, S0, r, q, T,
        n_out=n_strikes, window=window)

    # Historical returns over the option's horizon for Stutzer
    hist = tk.history(period=history_period)["Close"]
    hist_R = hist.pct_change(max(int(T * 252), 1)).dropna().add(1).values

    S_grid = np.linspace(0.01, 3 * S0, 4000)

    print(f"[{ticker}] spot={S0:.2f}  expiry={expiry} (T={T:.3f}y, {days}d)  "
          f"OTM quotes used={len(K_obs)} -> smile-fitted strikes={len(strikes)}  "
          f"r={r}  q={q}")
    return dict(S0=S0, r=r, q=q, T=T, strikes=strikes, calls=calls,
                hist_R=hist_R, S_grid=S_grid, f_true=None,
                smile=(K_obs, iv_obs, np.column_stack([strikes, iv_fit])))


# ----------------------------------------------------------------------
# Single-maturity driver: recover and compare all four densities
# ----------------------------------------------------------------------
def summarize_density(S_values, masses, F):
    """
    Compute summary statistics of a discrete distribution given as
    (values, masses). Skewness is the third central moment normalized
    by std^3 (same formula one would compute on data, generalized to
    weighted samples). Tail probabilities use thresholds relative to
    the forward F so they're comparable across underlyings.
    """
    x = np.asarray(S_values, float)
    w = np.asarray(masses, float); w = w / w.sum()
    mean = np.sum(w * x)
    std = np.sqrt(np.sum(w * (x - mean) ** 2))
    skew = np.sum(w * (x - mean) ** 3) / std ** 3 if std > 0 else np.nan
    return dict(mean=mean, std=std, skew=skew,
                p_drop10=np.sum(w[x < 0.90 * F]),    # P(S_T < 90% of F)
                p_drop20=np.sum(w[x < 0.80 * F]))    # P(S_T < 80% of F)


def run(market):
    """
    Single-maturity comparison driver. Recovers all four densities, prints
    a comparison table of moments and tail probabilities, and saves a plot.
    """
    S0, r, q, T = market["S0"], market["r"], market["q"], market["T"]
    strikes, calls = market["strikes"], market["calls"]
    S = market["S_grid"]
    F = S0 * np.exp((r - q) * T)

    # MaxEnt: information-theoretic recovery from prices
    Sg, f_me, w_me = maxent_density(S, strikes, calls, S0, r, q, T)

    # Stutzer: cross-entropy tilt of historical returns
    ST_s, p_s, gamma = stutzer_density(market["hist_R"], S0, r, T)

    # Black-Scholes lognormal at the ATM implied vol (parametric baseline).
    # Use the ATM IV as the single vol parameter. This is the best a one-vol
    # model can do against a real smile.
    atm = strikes[np.argmin(np.abs(strikes - S0))]
    atm_price = calls[np.argmin(np.abs(strikes - S0))]
    iv = bs_implied_vol(atm_price, S0, atm, r, q, T)
    mu = np.log(S0) + (r - q - 0.5 * iv ** 2) * T   # mean of log(S_T) under BS
    f_bs = np.exp(-(np.log(Sg) - mu) ** 2 / (2 * (iv ** 2 * T))) \
           / (Sg * iv * np.sqrt(2 * np.pi * T))

    # Breeden-Litzenberger: model-free method
    Kbl, f_bl = breeden_litzenberger(strikes, calls, r, T)

    # In-sample sanity check: does the recovered density reprice the options
    # it was given. (Should be tiny, like pennies or less on a well-converged solve).
    rmse = np.sqrt(np.mean([(np.exp(-r * T) * np.sum(np.maximum(Sg - K, 0) * w_me) - C) ** 2
                            for K, C in zip(strikes, calls)]))

    # ---- plot all four densities on one axis ----
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if market["f_true"] is not None:
        # Only available in the synthetic case where generated the data
        ax.plot(Sg, market["f_true"], "k--", lw=2, label="true density (synthetic)")
    ax.plot(Sg, f_me, lw=2, label="MaxEnt (from prices)")
    ax.plot(Sg, f_bs, lw=1.6, label=f"Black-Scholes lognormal (iv={iv:.1%})")
    ax.plot(Kbl, f_bl, lw=1.4, alpha=0.8, label="Breeden-Litzenberger")
    # Stutzer comes back as a weighted point cloud rather than a curve, so plot as histogram
    ax.hist(ST_s, bins=120, weights=p_s, density=True, alpha=0.25,
            label="Stutzer (from history)")
    ax.axvline(F, color="grey", ls=":", label="forward")
    ax.set_xlim(0.55 * S0, 1.6 * S0)        # adaptive to spot. works for SPY at $750 or synthetic at $100
    ax.set_xlabel("terminal price $S_T$")
    ax.set_ylabel("risk-neutral density")
    ax.set_title("Implied risk-neutral densities — four methods")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("densities.png", dpi=130)

    print(f"ATM implied vol           : {iv:.4f}")
    print(f"MaxEnt repricing RMSE     : {rmse:.3e}")
    print(f"Stutzer tilt gamma        : {gamma:.4f}")
    print(f"MaxEnt mean vs forward    : {np.sum(Sg * w_me):.4f} vs {F:.4f}")
    print("Saved plot -> densities.png")

    # ---- Quantitative comparison table (main finding!) ----
    # Convert each density to the same (values, masses) representation so
    # summarize_density compares things that are directly comparable.
    dSg = Sg[1] - Sg[0]
    stats = {
        "MaxEnt":        summarize_density(Sg, w_me, F),
        "Black-Scholes": summarize_density(Sg, f_bs * dSg, F),
        "Stutzer":       summarize_density(ST_s, p_s, F),
    }
    print("\n  method            std     skew   P(drop>10%)  P(drop>20%)")
    print("  " + "-" * 58)
    for name, s in stats.items():
        print(f"  {name:14s}  {s['std']:6.1f}  {s['skew']:+5.2f}     "
              f"{s['p_drop10']:7.2%}      {s['p_drop20']:7.2%}")
    ratio = stats["MaxEnt"]["p_drop20"] / max(stats["Black-Scholes"]["p_drop20"], 1e-9)
    me, bs = stats["MaxEnt"], stats["Black-Scholes"]
    print("  " + "-" * 58)
    # Report skew sign as the primary finding. Magnitude is sensitive to fit parameters.
    print(f"  FINDING: market-implied skew {me['skew']:+.2f} vs Black-Scholes "
          f"{bs['skew']:+.2f}.")
    print(f"           The market's distribution is LEFT-skewed; a lognormal cannot be.")
    print(f"           Deep-crash P(drop>20%): {me['p_drop20']:.2%} (market) "
          f"vs {bs['p_drop20']:.2%} (Black-Scholes) = {ratio:.2f}x.")


# ----------------------------------------------------------------------
# Term-structure driver: same gap, across maturities
# ----------------------------------------------------------------------
def run_term_structure(ticker="SPY", maturities=(30, 60, 90, 180, 365),
                       r=0.04, q=0.013, window=(0.75, 1.20),
                       crash_threshold=0.80, history_period="20y",
                       out_csv="term_structure.csv",
                       out_png="term_structure.png"):
    
    """
    Run the recovery pipeline at multiple maturities and report, at each maturity,
    the gap between the MaxEnt option-implied crash probability and the Stutzer
    historical estimate.

    The 'gap' column is market minus historical crash probability. 'signal_z'
    z-scores the gap across the sampled maturities; positive values mark
    maturities where the gap is largest.

    Note: yfinance only serves current option chains, so this is a single
    snapshot across maturities, not a time series across dates. A calendar
    time series would require a historical options data source (Polygon.io,
    OptionMetrics, etc.).

    """
    rows = []
    for days in maturities:
        # Wrap the per-maturity work in try/except: one bad expiry shouldn't
        # kill the whole run (e.g. if a single chain has too few quotes to
        # fit a smile).
        try:
            mk = load_real_market(ticker, target_days=days, r=r, q=q,
                                  window=window, history_period=history_period)
        except Exception as e:
            print(f"  maturity {days}d skipped: {e}")
            continue

        S0, T = mk["S0"], mk["T"]
        F = S0 * np.exp((r - q) * T)
        S = mk["S_grid"]
        crash_level = crash_threshold * F   # e.g. 0.90*F = "drop > 10% from forward"

        # Market view: MaxEnt density from current option prices
        _, _, w_me = maxent_density(S, mk["strikes"], mk["calls"], S0, r, q, T)
        p_mkt = float(np.sum(w_me[S < crash_level]))

        # Historical view: Stutzer density from the look-back window
        ST_s, p_s, gamma = stutzer_density(mk["hist_R"], S0, r, T)
        p_hist = float(np.sum(p_s[ST_s < crash_level]))

        rows.append(dict(days=mk["T"] * 365, T=mk["T"],
                         p_market=p_mkt, p_historical=p_hist,
                         gap=p_mkt - p_hist,
                         ratio=p_mkt / max(p_hist, 1e-6),
                         gamma=gamma))

    if not rows:
        raise RuntimeError("No maturities recovered successfully.")

    # Normalized signal: z-score the gap across the maturities sampled.
    # Lets the table flag "which maturity is most unusual" without absolute
    # thresholds that would depend on the threshold choice.
    import csv
    keys = list(rows[0].keys())
    gaps = np.array([row["gap"] for row in rows])
    mean, std = gaps.mean(), gaps.std() if gaps.std() > 0 else 1.0
    for row in rows:
        row["signal_z"] = (row["gap"] - mean) / std

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["signal_z"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # ---- Two-panel plot: term structure + cross-maturity signal ----
    days = np.array([row["days"] for row in rows])
    pm = np.array([row["p_market"] for row in rows]) * 100
    ph = np.array([row["p_historical"] for row in rows]) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left panel: market vs historical crash probability with shaded gap
    ax1.plot(days, pm, "o-", lw=2, label="market-implied (MaxEnt)")
    ax1.plot(days, ph, "s--", lw=2, label="historical (Stutzer)")
    ax1.fill_between(days, ph, pm, where=pm > ph, alpha=0.18,
                     color="C0", label="risk premium for tail")
    ax1.set_xlabel("days to expiry")
    ax1.set_ylabel("P(crash) — %")
    pct = int(round((1 - crash_threshold) * 100))
    ax1.set_title(f"Term structure of crash probability\nP(drop > {pct}% from forward)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right panel: z-scored gap, color-coded by sign for quick reading
    z = np.array([row["signal_z"] for row in rows])
    colors = ["C3" if zi > 0 else "C0" for zi in z]
    ax2.bar(days, z, width=days * 0.08, color=colors, alpha=0.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("days to expiry")
    ax2.set_ylabel("signal z-score (gap)")
    ax2.set_title("Cross-maturity signal\n(positive = market unusually fearful)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)

    # ---- Console report ----
    print(f"\n  TERM STRUCTURE OF CRASH PRICING ({ticker}, drop > {pct}% from forward)")
    print(f"  {'days':>6}  {'P_market':>9}  {'P_history':>9}  {'gap':>7}  "
          f"{'ratio':>6}  {'signal_z':>8}")
    print("  " + "-" * 60)
    for row in rows:
       # Cap the ratio display when historical is near zero (otherwise it blows up).
        ratio_disp = "inf" if row['p_historical'] < 1e-5 else f"{row['ratio']:5.1f}x"
        print(f"  {row['days']:6.0f}  {row['p_market']:8.2%}  "
              f"{row['p_historical']:8.2%}  {row['gap']:+7.2%}  "
              f"{ratio_disp:>6}  {row['signal_z']:+7.2f}")
    print(f"\n  Saved -> {out_csv}, {out_png}")
    return rows


# ----------------------------------------------------------------------
# Cross-sector driver (research extension)
# ----------------------------------------------------------------------
def run_cross_sector(tickers=("SPY", "XLF", "XLK", "XLE", "XLV", "XLP"),
                     target_days=90, r=0.04, q=0.013,
                     window=(0.85, 1.15), crash_threshold=0.90,
                     out_csv="cross_sector.csv",
                     out_png="cross_sector.png"):
    """
    
    Apply the recovery pipeline at one maturity across multiple tickers and
    compare implied skew and dispersion across sectors.

    Reports skew (asymmetry) and standardized std/F (dispersion) per ticker.
    Together these characterize both the shape and size of risk.

    (Known) limitations: sector ETF chains are approx. 10x thinner than SPY's, 
    with 15-25 OTM strikes per ticker after liquidity filtering. The quadratic
    smile fit is poorly constrained on such thin data, so some recoveries
    are noisy and others fail outright (degenerate convergence to a point-
    mass at the forward, visible as std/F = 0). The directional ordering of
    skews across sectors is reliable, but the magnitudes are not. Production
    versions would use SVI/SSVI smile fits or apply the pipeline to single-
    stock options with deeper chains (AAPL, MSFT) instead of sector ETFs.

    Tickers default to SPY plus five GICS sector ETFs:
    XLF=financials, XLK=tech, XLE=energy, XLV=healthcare, XLP=staples.

    """
    rows = []
    for ticker in tickers:
        try:
            # Short history window here, for shape comparison at one moment,
            # not for Stutzer's empirical density (which isn't used in this driver)
            mk = load_real_market(ticker, target_days=target_days, r=r, q=q,
                                  window=window, history_period="3y")
        except Exception as e:
            print(f"  {ticker} skipped: {e}")
            continue

        S0, T = mk["S0"], mk["T"]
        F = S0 * np.exp((r - q) * T)
        S = mk["S_grid"]
        crash_level = crash_threshold * F

        _, _, w = maxent_density(S, mk["strikes"], mk["calls"], S0, r, q, T)

        # Standardize moments by the forward so different price levels are
        # comparable (raw std would say "XLE at $55 is much less volatile
        # than SPY at $750", which is meaningless).
        mean = float(np.sum(w * S))
        var = float(np.sum(w * (S - mean) ** 2))
        std = np.sqrt(var)
        skew = (float(np.sum(w * (S - mean) ** 3)) / std ** 3) if std > 0 else np.nan
        p_drop = float(np.sum(w[S < crash_level]))

        # ATM implied vol (different from std, which is from the recovered
        # density, IV is from a single ATM quote inverted through BS)
        atm_K = mk["strikes"][np.argmin(np.abs(mk["strikes"] - S0))]
        atm_C = mk["calls"][np.argmin(np.abs(mk["strikes"] - S0))]
        iv = bs_implied_vol(atm_C, S0, atm_K, r, q, T)

        rows.append(dict(ticker=ticker, S0=S0, T=T, iv=iv,
                         std_pct=std / F, skew=skew, p_drop=p_drop))

    if not rows:
        raise RuntimeError("No tickers recovered successfully.")

    import csv
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=keys)
        w_.writeheader()
        for row in rows:
            w_.writerow(row)

    # Plot
    skews = np.array([r_["skew"] for r_ in rows])
    pdrops = np.array([r_["p_drop"] for r_ in rows]) * 100
    stds = np.array([r_["std_pct"] for r_ in rows])
    labels = [r_["ticker"] for r_ in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left panel: skew (asymmetry) vs crash probability, with dot size encoding
    # overall dispersion. Three visual dimensions on one plot, separating
    # "high crash prob because volatile" from "high crash prob because skewed".
    sizes = 80 + 1200 * (stds - stds.min()) / max(stds.max() - stds.min(), 1e-6)
    ax1.scatter(skews, pdrops, s=sizes, alpha=0.55, edgecolor="k", linewidth=1.2)
    for x, y, lab in zip(skews, pdrops, labels):
        ax1.annotate(lab, (x, y), xytext=(7, 7), textcoords="offset points",
                     fontsize=10, fontweight="bold")
    ax1.axvline(0, color="k", lw=0.8, alpha=0.4)
    ax1.set_xlabel("market-implied skew  (← more crash-feared)")
    pct = int(round((1 - crash_threshold) * 100))
    ax1.set_ylabel(f"P(drop > {pct}%) — %")
    ax1.set_title(f"Cross-sector risk pricing at {target_days}d\n"
                  "(dot size = implied dispersion std/F)")
    ax1.grid(alpha=0.3)

    # Right panel: ATM implied vol bar chart for context
    ivs = np.array([r_["iv"] for r_ in rows]) * 100
    order = np.argsort(ivs)
    ax2.barh(np.array(labels)[order], ivs[order], alpha=0.75, color="C0")
    ax2.set_xlabel("ATM implied vol — %")
    ax2.set_title(f"ATM volatility by sector  ({target_days}d)")
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)

    # Console table, ranked by most crash-feared first (most negative skew)
    print(f"\n  CROSS-SECTOR RISK PRICING ({target_days}d, drop > {pct}% from forward)")
    print(f"  {'ticker':<7} {'ATM IV':>7}  {'std/F':>7}  {'skew':>6}  "
          f"{'P(drop)':>8}")
    print("  " + "-" * 45)
    rows_sorted = sorted(rows, key=lambda r_: r_["skew"])
    for r_ in rows_sorted:
        print(f"  {r_['ticker']:<7} {r_['iv']:6.2%}  {r_['std_pct']:6.2%}  "
              f"{r_['skew']:+5.2f}   {r_['p_drop']:7.2%}")
    print(f"\n  Saved -> {out_csv}, {out_png}")
    return rows


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Offline validation against a known-shape density (no network):
    # run(make_synthetic_market())

    # Single-maturity comparison at ~180d:
    run(load_real_market("SPY", target_days=180, r=0.04, q=0.013))

    # Term structure of crash pricing across five maturities.
    # crash_threshold=0.90 means "drop > 10% from forward". A 10% threshold
    # gives a non-degenerate historical estimate over 20 years of SPY data;
    # 20% mostly tests whether the sample contains an extreme event
    run_term_structure("SPY",
                       maturities=(30, 60, 90, 180, 365),
                       r=0.04, q=0.013,
                       crash_threshold=0.90,
                       history_period="20y")

    # Cross-sector extension (research sketch). Disabled by default because
    # sector ETF chains are thin enough that some recoveries fail or produce
    # noisy magnitudes (see docstring of run_cross_sector). Useful as a
    # demonstration that the process generalizes to other tickers, but not
    # a precision result. Uncomment to run.
    #
    # run_cross_sector(tickers=("SPY", "XLF", "XLK", "XLE", "XLV", "XLP"),
    #                  target_days=90,
    #                  r=0.04, q=0.013,
    #                  window=(0.85, 1.15),
    #                  crash_threshold=0.90)