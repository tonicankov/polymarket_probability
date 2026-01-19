import random
import math


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_truncated_normal(mean: float, std: float, lo: float, hi: float) -> float:
    # Simple rejection sampling (fine for this scale)
    while True:
        x = random.gauss(mean, std)
        if lo <= x <= hi:
            return x


def kelly_fraction_yes(p: float, q: float) -> float:
    """
    Full Kelly fraction for buying YES at price q (implied probability).
    f* = (p - q) / (1 - q), clipped at 0.
    """
    if q >= 1.0:
        return 0.0
    f = (p - q) / (1.0 - q)
    return max(0.0, f)


def run_one_trial(
    n_bets: int,
    bankroll0: float,
    alpha: float,          # 1.0 full, 0.5 half, 0.25 quarter
    f_cap: float,          # max fraction of bankroll per bet
    q_mean: float,
    q_std: float,
    q_lo: float,
    q_hi: float,
    edge_frac_mean: float, # edge as fraction of (1-q)
    edge_frac_std: float,
    edge_frac_lo: float,
    edge_frac_hi: float,
    min_edge_frac_to_bet: float,
) -> float:
    bankroll = bankroll0

    for _ in range(n_bets):
        # Market implied probability q in [q_lo, q_hi], centered around ~0.5
        q = sample_truncated_normal(q_mean, q_std, q_lo, q_hi)

        # Your edge is modeled as a fraction of the remaining probability mass (1 - q)
        # so p = q + edge_frac*(1-q) never exceeds 1.
        edge_frac = sample_truncated_normal(edge_frac_mean, edge_frac_std, edge_frac_lo, edge_frac_hi)

        # Only bet if edge is at least threshold (as fraction of 1-q)
        if edge_frac < min_edge_frac_to_bet:
            continue

        p = q + edge_frac * (1.0 - q)  # guaranteed <= 1

        # Kelly sizing for YES at q
        f = alpha * kelly_fraction_yes(p, q)
        f = clamp(f, 0.0, f_cap)
        if f <= 0.0:
            continue

        stake = bankroll * f

        # Payout for buying YES at price q (ignoring fees):
        # Win profit = stake*(1-q)/q ; loss = stake
        b = (1.0 - q) / q

        win = random.random() < p  # in this sim, p is treated as the true probability
        if win:
            bankroll += stake * b
        else:
            bankroll -= stake

        if bankroll <= 0.0:
            return 0.0

    return bankroll


def main():
    random.seed(1)

    # Simulation controls
    n_bets = 100
    n_trials = 1000
    bankroll0 = 1000.0

    # Kelly controls
    alpha = 0.5     # half Kelly
    f_cap = 0.25    # cap bet fraction

    # q distribution: truncated normal in [0.2, 0.8], mean 0.5
    q_lo, q_hi = 0.20, 0.80
    q_mean = 0.50
    q_std = 0.12    # adjust if you want tighter around 0.5

    # Edge model:
    # edge_frac in [0.10, 0.60], mean 0.20.
    # Interpreted as: p = q + edge_frac*(1-q).
    # Example: q=0.75 and edge_frac=0.50 => p = 0.75 + 0.5*0.25 = 0.875 (not 1.30)
    edge_frac_lo, edge_frac_hi = 0.10, 0.60
    edge_frac_mean = 0.20
    edge_frac_std = 0.10

    # Only bet if edge_frac >= threshold
    min_edge_frac_to_bet = 0.10

    results = []
    for _ in range(n_trials):
        results.append(
            run_one_trial(
                n_bets=n_bets,
                bankroll0=bankroll0,
                alpha=alpha,
                f_cap=f_cap,
                q_mean=q_mean,
                q_std=q_std,
                q_lo=q_lo,
                q_hi=q_hi,
                edge_frac_mean=edge_frac_mean,
                edge_frac_std=edge_frac_std,
                edge_frac_lo=edge_frac_lo,
                edge_frac_hi=edge_frac_hi,
                min_edge_frac_to_bet=min_edge_frac_to_bet,
            )
        )

    avg_final = sum(results) / len(results)
    results_sorted = sorted(results)
    median_final = results_sorted[len(results_sorted) // 2]
    ruin_rate = sum(1 for x in results if x == 0.0) / len(results)

    print("avg final:", avg_final)
    print("median final:", median_final)
    print("ruin rate:", ruin_rate)


if __name__ == "__main__":
    main()