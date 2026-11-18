#!/usr/bin/env python3
"""
الاختبار التاريخي لإشارات الماسح
---------------------------------
يطبّق نفس قواعد إشارة الدخول في scanner.py على كل يوم خلال الفترة المحددة، ثم يقيس:
  1) هل تحرك السهم في اتجاه الإشارة بعد 1 و3 و5 أيام تداول؟
  2) هل وصل السهم لنقطة تعادل عقد أسبوعي (دلتا 0.40) قبل الانتهاء؟
  3) محاكاة صفقة العقد: شراء عند إغلاق يوم الإشارة، جني ربح / وقف خسارة، أو خروج قبل الانتهاء.

لأن أسعار العقود التاريخية غير متوفرة مجاناً، يُسعَّر العقد بمعادلة بلاك-شولز
باستخدام التذبذب التاريخي × معامل (iv_multiplier). النتائج تقدير وليست أداءً حقيقياً.

التشغيل:
    python backtest.py                       # السوق كامل، آخر سنتين
    python backtest.py --years 3
    python backtest.py --tickers NVDA TSLA AMD
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import scanner as sc

BASE = Path(__file__).resolve().parent
log = logging.getLogger("backtest")
N = NormalDist()

DEFAULTS = {
    "iv_multiplier": 1.2,      # التذبذب الضمني عادة أعلى من التاريخي
    "min_iv": 0.15,
    "dte_days": 7,             # عقد أسبوعي (أيام تقويمية)
    "take_profit": 0.50,       # +50% على العقد
    "stop_loss": 0.50,         # -50% على العقد
    "exit_days_before_expiry": 1,
    "cooldown_days": 5,        # لا تُحسب إشارة جديدة لنفس السهم خلال 5 أيام تداول
}


# ============================================================
# تسعير العقد
# ============================================================

def bs_price(S: float, K: float, T: float, sigma: float, r: float, call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / sq
    d2 = d1 - sq
    if call:
        return S * N.cdf(d1) - K * math.exp(-r * T) * N.cdf(d2)
    return K * math.exp(-r * T) * N.cdf(-d2) - S * N.cdf(-d1)


def strike_for_delta(S: float, T: float, sigma: float, r: float, delta: float, call: bool) -> float:
    d1 = N.inv_cdf(delta) if call else N.inv_cdf(1 - delta)
    return S * math.exp(-(d1 * sigma * math.sqrt(T) - (r + sigma ** 2 / 2) * T))


def simulate_trade(f: pd.DataFrame, i: int, bias: str, bt: dict, opt: dict) -> dict | None:
    """يحاكي شراء عقد أسبوعي عند إغلاق اليوم i."""
    call = bias == "bull"
    sign = 1 if call else -1
    S0 = float(f["close"].iloc[i])
    hv = float(f["hv"].iloc[i])
    if not hv or np.isnan(hv):
        return None
    sigma = max(hv * bt["iv_multiplier"], bt["min_iv"])
    r = opt["risk_free_rate"]
    d0 = pd.Timestamp(f.index[i]).date()
    expiry = d0 + timedelta(days=bt["dte_days"])
    T0 = bt["dte_days"] / 365
    K = strike_for_delta(S0, T0, sigma, r, opt["target_delta"], call)
    P0 = bs_price(S0, K, T0, sigma, r, call)
    if P0 <= 0.01:
        return None
    be = K + sign * P0
    em = S0 * sigma * math.sqrt(T0)

    # مسار الأيام التالية حتى الانتهاء
    exit_ret, exit_reason, exit_day = None, "", None
    touched_be, mfe = False, 0.0
    last_hold = expiry - timedelta(days=bt["exit_days_before_expiry"])
    j = i + 1
    while j < len(f):
        dj = pd.Timestamp(f.index[j]).date()
        if dj > expiry:
            break
        hi, lo, cl = float(f["high"].iloc[j]), float(f["low"].iloc[j]), float(f["close"].iloc[j])
        fav = (hi - S0) if call else (S0 - lo)
        mfe = max(mfe, fav)
        if (call and hi >= be) or (not call and lo <= be):
            touched_be = True
        if exit_ret is None:
            T = max((expiry - dj).days, 0) / 365
            v = bs_price(cl, K, T, sigma, r, call)
            ret = v / P0 - 1
            if ret >= bt["take_profit"]:
                exit_ret, exit_reason, exit_day = ret, "جني ربح", j - i
            elif ret <= -bt["stop_loss"]:
                exit_ret, exit_reason, exit_day = ret, "وقف خسارة", j - i
            elif dj >= last_hold:
                exit_ret, exit_reason, exit_day = ret, "خروج زمني", j - i
        j += 1
    if exit_ret is None:
        return None   # الإشارة حديثة ولم تكتمل مدتها

    def fwd(n):
        k = i + n
        return sign * (float(f["close"].iloc[k]) / S0 - 1) if k < len(f) else np.nan

    return {
        "date": d0.isoformat(), "bias": bias, "price": round(S0, 2), "score": float(f["score"].iloc[i]),
        "compressed": bool(f["compressed"].iloc[i]), "rvol": round(float(f["rvol"].iloc[i]), 2),
        "strike": round(K, 2), "premium": round(P0, 2), "breakeven": round(be, 2),
        "exp_move_pct": round(em / S0 * 100, 2),
        "ret_1d": fwd(1), "ret_3d": fwd(3), "ret_5d": fwd(5),
        "touched_breakeven": touched_be, "mfe_vs_em": round(mfe / em, 2) if em else np.nan,
        "option_ret": round(exit_ret, 4), "exit": exit_reason, "hold_days": exit_day,
    }


# ============================================================
# التشغيل والتقرير
# ============================================================

def run_ticker(sym: str, df: pd.DataFrame, cfg: dict, bt: dict, start: date) -> list[dict]:
    if len(df.dropna(subset=["Close"])) < 80:
        return []
    f = sc.compute_frame(df, cfg)
    mask = (f["entry"] & f["ready"] & f["liquid"] & (f["bias"] != "neutral")
            & (f["score"] >= cfg["signals"]["min_setup_score"]))
    trades, last_i = [], -10 ** 9
    for i in np.flatnonzero(mask.values):
        if pd.Timestamp(f.index[i]).date() < start or i - last_i < bt["cooldown_days"]:
            continue
        t = simulate_trade(f, i, f["bias"].iloc[i], bt, cfg["options"])
        if t:
            t["symbol"] = sym
            trades.append(t)
            last_i = i
    return trades


def baseline(data: dict[str, pd.DataFrame], start: date) -> tuple[float, float]:
    """نسبة الأيام العادية اللي ارتفع فيها السهم بعد 5 أيام — للمقارنة."""
    ups = tot = 0
    for df in data.values():
        c = df["Close"].dropna()
        c = c[c.index >= pd.Timestamp(start)]
        r5 = (c.shift(-5) / c - 1).dropna()
        ups += int((r5 > 0).sum()); tot += len(r5)
    up = ups / tot if tot else float("nan")
    return up, 1 - up


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    o = t["option_ret"]
    wins, losses = o[o > 0], o[o <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() else float("inf")
    return {
        "n": len(t),
        "dir_1d": (t["ret_1d"] > 0).mean(), "dir_3d": (t["ret_3d"] > 0).mean(), "dir_5d": (t["ret_5d"] > 0).mean(),
        "be": t["touched_breakeven"].mean(),
        "win": (o > 0).mean(), "avg": o.mean(), "med": o.median(),
        "avg_win": wins.mean() if len(wins) else 0.0, "avg_loss": losses.mean() if len(losses) else 0.0,
        "pf": pf,
    }


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.0f}%"


def report(trades: pd.DataFrame, base: tuple[float, float], years: float, n_tickers: int, bt: dict) -> str:
    s = stats(trades)
    if not s:
        return "📊 الاختبار التاريخي\nلا توجد إشارات دخول في الفترة المحددة."
    L = [
        f"📊 الاختبار التاريخي — آخر {years:g} سنة | {n_tickers} سهم",
        f"عدد الإشارات: {s['n']:,} (حوالي {s['n'] / max(years * 52, 1):.1f} أسبوعياً)",
        "",
        "اتجاه السهم بعد الإشارة (صح):",
        f"  بعد يوم {pct(s['dir_1d'])} | 3 أيام {pct(s['dir_3d'])} | 5 أيام {pct(s['dir_5d'])}",
        f"  للمقارنة: أي يوم عشوائي — صعود {pct(base[0])} / هبوط {pct(base[1])} بعد 5 أيام",
        f"وصل السهم لنقطة تعادل العقد قبل الانتهاء: {pct(s['be'])}",
        "",
        f"محاكاة العقد (دلتا 0.40، ربح +{bt['take_profit']:.0%} / وقف -{bt['stop_loss']:.0%}):",
        f"  نسبة الصفقات الرابحة: {pct(s['win'])}",
        f"  متوسط العائد للصفقة: {s['avg'] * 100:+.1f}% (الوسيط {s['med'] * 100:+.1f}%)",
        f"  متوسط الربح {s['avg_win'] * 100:+.1f}% | متوسط الخسارة {s['avg_loss'] * 100:+.1f}%",
        f"  معامل الربح: {s['pf']:.2f}  (أعلى من 1 = الأرباح أكبر من الخسائر)",
        "",
        "التفصيل:",
    ]

    def line(name, sub):
        x = stats(sub)
        if x:
            L.append(f"  {name}: {x['n']} إشارة | رابحة {pct(x['win'])} | متوسط {x['avg'] * 100:+.1f}% | اتجاه 5 أيام {pct(x['dir_5d'])}")

    line("صعود (كول)", trades[trades["bias"] == "bull"])
    line("هبوط (بوت)", trades[trades["bias"] == "bear"])
    line("مع انضغاط", trades[trades["compressed"]])
    line("بدون انضغاط", trades[~trades["compressed"]])
    line("تقييم 45–60", trades[trades["score"] < 60])
    line("تقييم 60–75", trades[(trades["score"] >= 60) & (trades["score"] < 75)])
    line("تقييم 75+", trades[trades["score"] >= 75])
    yrs = pd.to_datetime(trades["date"]).dt.year
    for y in sorted(yrs.unique()):
        line(f"سنة {y}", trades[yrs == y])

    L += [
        "",
        "⚠️ تقدير: أسعار العقود محسوبة بمعادلة بلاك-شولز (التذبذب التاريخي ×"
        f" {bt['iv_multiplier']}) والدخول عند إغلاق يوم الإشارة. الأداء الحقيقي يتأثر بفارق"
        " bid/ask وقفزات التذبذب الضمني حول الأرباح وتأخر البيانات. لا يتضمن الاختبار نقاط"
        " تدفق العقود والأرباح لأنها غير متوفرة تاريخياً.",
    ]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="اختبار تاريخي لإشارات الماسح")
    ap.add_argument("--years", type=float, default=2)
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--config", default=str(BASE / "config.json"))
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    cfg = sc.load_config(Path(args.config))
    bt = {**DEFAULTS, **cfg.get("backtest", {})}
    tickers = [t.upper() for t in args.tickers] if args.tickers else sc.load_universe(cfg)
    start = date.today() - timedelta(days=int(args.years * 365))
    period = f"{int(math.ceil(args.years)) + 1}y"

    log.info("تحميل %s من البيانات لـ %d سهم…", period, len(tickers))
    data = sc.download_daily(tickers, period=period)
    log.info("تم تحميل %d سهم — بدء الاختبار", len(data))

    rows = []
    for k, (sym, df) in enumerate(data.items(), 1):
        try:
            rows += run_ticker(sym, df, cfg, bt, start)
        except Exception as e:
            log.debug("%s: %s", sym, e)
        if k % 100 == 0:
            log.info("… %d/%d", k, len(data))

    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades = trades.sort_values("date")
        trades.to_csv(BASE / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    text = report(trades, baseline(data, start), args.years, len(data), bt)
    (BASE / "backtest_summary.txt").write_text(text, encoding="utf-8")
    if args.no_telegram:
        print(text)
    else:
        sc.send_telegram(cfg, text)


if __name__ == "__main__":
    main()
