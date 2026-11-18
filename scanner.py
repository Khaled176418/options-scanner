#!/usr/bin/env python3
"""
ماسح الأسهم الأمريكية والعقود الأسبوعية
----------------------------------------
- يمسح السوق ويبحث عن أسهم مرشحة لحركة حادة (انضغاط تذبذب + اختراق + حجم غير اعتيادي + نشاط عقود).
- يرشّح عقد أسبوعي (سترايك + انتهاء) حسب الدلتا والسيولة والحركة المتوقعة.
- يرسل تنبيه تيليجرام: قائمة مرشحين يومية + تنبيه دخول لحظي عند تحقق شرط الاختراق.

التشغيل:
    python scanner.py                 # تشغيل مستمر وقت السوق
    python scanner.py --once          # مسح واحد ثم خروج
    python scanner.py --test-telegram # اختبار وصول الرسائل
    python scanner.py --tickers NVDA TSLA AMD --once   # مسح أسهم محددة
    python backtest.py                # اختبار تاريخي لنفس القواعد

تنبيه: بيانات Yahoo Finance متأخرة ~15 دقيقة. الأداة للمساعدة في الفرز وليست توصية استثمارية.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:  # يسمح بتشغيل الاختبارات بدون yfinance
    yf = None

BASE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
RIYADH = ZoneInfo("Asia/Riyadh")
UA = {"User-Agent": "Mozilla/5.0 (options-scanner)"}

log = logging.getLogger("scanner")

FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "ORCL", "CRM", "ADBE", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
    "PLTR", "SNOW", "CRWD", "PANW", "NOW", "SHOP", "UBER", "ABNB", "DDOG", "NET",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA", "PYPL", "AXP",
    "XOM", "CVX", "OXY", "COP", "SLB", "LLY", "UNH", "JNJ", "PFE", "MRK",
    "ABBV", "WMT", "COST", "HD", "TGT", "NKE", "SBUX", "MCD", "DIS", "BA",
    "CAT", "DE", "GE", "LMT", "RTX", "F", "GM", "SPY", "QQQ", "IWM",
    "COIN", "MSTR", "HOOD", "SOFI", "RIVN", "MARA", "SMCI", "ARM", "RDDT", "IONQ",
]


# ============================================================
# الإعدادات والحالة
# ============================================================

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    # أسرار GitHub (أو متغيرات البيئة) لها الأولوية على الملف
    tg = cfg.setdefault("telegram", {})
    tg["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN") or tg.get("bot_token", "")
    tg["chat_id"] = os.getenv("TELEGRAM_CHAT_ID") or tg.get("chat_id", "")
    return cfg


class State:
    """يحفظ التنبيهات المرسلة حتى لا تتكرر في نفس اليوم."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}
        today = datetime.now(ET).date().isoformat()
        if self.data.get("day") != today:
            self.data = {"day": today, "sent": []}

    def seen(self, key: str) -> bool:
        return key in self.data["sent"]

    def mark(self, key: str):
        self.data["sent"].append(key)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


# ============================================================
# قائمة الأسهم
# ============================================================

def _wiki_tickers(url: str, col_candidates: list[str]) -> list[str]:
    html = requests.get(url, headers=UA, timeout=20).text
    for table in pd.read_html(StringIO(html)):
        for col in col_candidates:
            if col in table.columns:
                return [str(s).strip().replace(".", "-") for s in table[col].dropna()]
    return []


def load_universe(cfg: dict) -> list[str]:
    cache = BASE / "universe_cache.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        return json.loads(cache.read_text())

    u = cfg["universe"]
    tickers: set[str] = set(u.get("extra_tickers", []))
    try:
        if u.get("use_sp500"):
            tickers |= set(_wiki_tickers(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"]))
        if u.get("use_nasdaq100"):
            tickers |= set(_wiki_tickers(
                "https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker", "Symbol"]))
    except Exception as e:
        log.warning("تعذر جلب قائمة الأسهم من ويكيبيديا (%s) — سأستخدم القائمة الاحتياطية", e)

    if len(tickers) < 50:
        tickers |= set(FALLBACK_UNIVERSE)
    out = sorted(tickers)
    cache.write_text(json.dumps(out))
    return out


# ============================================================
# المؤشرات الفنية
# ============================================================

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def session_fraction(now_et: datetime) -> float:
    """نسبة الجلسة المنقضية — لتصحيح الحجم النسبي أثناء اليوم."""
    if now_et.weekday() >= 5:
        return 1.0
    open_ = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if now_et <= open_ or now_et >= close:
        return 1.0
    return max(0.1, (now_et - open_).total_seconds() / (close - open_).total_seconds())


@dataclass
class Setup:
    symbol: str
    price: float
    change_pct: float
    score: float
    bias: str               # "bull" | "bear" | "neutral"
    entry: bool             # تحقق شرط الدخول الآن؟
    up_trigger: float
    down_trigger: float
    atr: float
    rvol: float
    hv: float               # التذبذب التاريخي السنوي
    bar_date: date | None = None
    reasons: list[str] = field(default_factory=list)


def compute_frame(df: pd.DataFrame, cfg: dict, sess_frac: float = 1.0) -> pd.DataFrame:
    """يحسب الإشارات لكل يوم باستخدام بيانات ذلك اليوم وما قبله فقط.
    نفس الدالة يستخدمها المسح الحي (آخر صف) والاختبار التاريخي (كل الصفوف)."""
    df = df.dropna(subset=["Close"]).copy()
    sig = cfg["signals"]
    lb = sig["breakout_lookback_days"]
    close, vol, high, low = df["Close"], df["Volume"].fillna(0), df["High"], df["Low"]

    ema20, ema50 = close.ewm(span=20).mean(), close.ewm(span=50).mean()
    sma20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    a = atr(df)
    bw = (4 * std20) / sma20
    squeeze = (sma20 + 2 * std20 < ema20 + 1.5 * a) & (sma20 - 2 * std20 > ema20 - 1.5 * a)
    bw_rank = bw.rolling(125, min_periods=21).apply(lambda x: float((x[:-1] < x[-1]).mean()), raw=True)

    avg_vol = vol.shift(1).rolling(20).mean()
    avg_eff = avg_vol.copy()
    if len(avg_eff):
        avg_eff.iloc[-1] *= sess_frac           # تصحيح حجم اليوم الجاري
    rvol = (vol / avg_eff.replace(0, np.nan)).fillna(0)

    hi = high.shift(1).rolling(lb).max()
    lo = low.shift(1).rolling(lb).min()
    atr_prev = a.shift(1)
    r = rsi(close)

    f = pd.DataFrame(index=df.index)
    f["close"], f["high"], f["low"] = close, high, low
    f["chg"] = close.pct_change() * 100
    f["hv"] = np.log(close).diff().rolling(20).std() * math.sqrt(252)
    f["atr"], f["rvol"], f["up"], f["dn"] = atr_prev, rvol, hi, lo
    f["squeeze"] = squeeze.shift(1).fillna(False).astype(bool)
    f["bw_rank"] = bw_rank.shift(1)
    f["compressed"] = f["squeeze"] | (f["bw_rank"] <= sig["squeeze_percentile"])
    f["broke_up"] = close > hi
    f["broke_dn"] = (close < lo) & ~f["broke_up"]
    dist = np.minimum((hi - close).abs(), (close - lo).abs()) / atr_prev
    f["near"] = ~f["broke_up"] & ~f["broke_dn"] & (dist < 0.5)
    f["expand"] = (high - low) / atr_prev > 1.5
    trend_up = (close > ema20) & (ema20 > ema50)
    trend_dn = (close < ema20) & (ema20 < ema50)

    f["score"] = (25 * f["compressed"] + 30 * (f["broke_up"] | f["broke_dn"]) + 10 * f["near"]
                  + rvol.clip(upper=4) / 4 * 25 + 10 * f["expand"].fillna(False)
                  + 5 * (trend_up | trend_dn)).round(1)
    bull = 3 * f["broke_up"] + trend_up + (r > 55)
    bear = 3 * f["broke_dn"] + trend_dn + (r < 45)
    f["bias"] = np.where(bull > bear, "bull", np.where(bear > bull, "bear", "neutral"))
    f["entry"] = (((f["broke_up"] & (f["bias"] == "bull")) | (f["broke_dn"] & (f["bias"] == "bear")))
                  & (rvol >= sig["min_rvol_for_entry"]))
    fl = cfg["filters"]
    f["liquid"] = (close >= fl["min_price"]) & (close * avg_vol >= fl["min_avg_dollar_volume"])
    f["ready"] = np.arange(len(f)) >= 59      # يحتاج 60 يوم على الأقل
    return f


def reasons_for(row: pd.Series, lb: int) -> list[str]:
    out = []
    if row["compressed"]:
        out.append("انضغاط بولنجر داخل كلتنر" if row["squeeze"]
                   else f"انضغاط تذبذب (أضيق {row['bw_rank']:.0%} خلال 6 أشهر)")
    if row["broke_up"]:
        out.append(f"اختراق قمة {lb} يوم ({row['up']:.2f})")
    elif row["broke_dn"]:
        out.append(f"كسر قاع {lb} يوم ({row['dn']:.2f})")
    elif row["near"]:
        out.append("قريب جداً من مستوى الاختراق")
    if row["rvol"] >= 1.5:
        out.append(f"حجم {row['rvol']:.1f}x المعدل")
    if row["expand"]:
        out.append("توسّع المدى اليومي")
    return out


def analyze_stock(sym: str, df: pd.DataFrame, cfg: dict, sess_frac: float = 1.0) -> Setup | None:
    if len(df.dropna(subset=["Close"])) < 60:
        return None
    f = compute_frame(df, cfg, sess_frac)
    row = f.iloc[-1]
    if not row["liquid"] or np.isnan(row["atr"]):
        return None
    return Setup(sym, float(row["close"]), float(row["chg"]), float(row["score"]), str(row["bias"]),
                 bool(row["entry"]), float(row["up"]), float(row["dn"]), float(row["atr"]),
                 float(row["rvol"]), float(row["hv"]), pd.Timestamp(f.index[-1]).date(),
                 reasons_for(row, cfg["signals"]["breakout_lookback_days"]))


def download_daily(tickers: list[str], chunk: int = 100, period: str = "8mo") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k in range(0, len(tickers), chunk):
        batch = tickers[k:k + chunk]
        try:
            data = yf.download(batch, period=period, interval="1d", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            log.warning("فشل تحميل دفعة %d: %s", k, e)
            continue
        for s in batch:
            try:
                df = data[s] if isinstance(data.columns, pd.MultiIndex) else data
                if not df.dropna(how="all").empty:
                    out[s] = df.dropna(how="all")
            except KeyError:
                pass
    return out


# ============================================================
# العقود
# ============================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(S: float, K: float, T: float, sigma: float, r: float, call: bool) -> float:
    if sigma <= 0 or T <= 0:
        return (1.0 if S > K else 0.0) if call else (-1.0 if S < K else 0.0)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if call else norm_cdf(d1) - 1


@dataclass
class Pick:
    kind: str
    contract: str
    strike: float
    expiry: str
    dte: int
    mid: float
    bid: float
    ask: float
    delta: float
    iv: float
    oi: int
    volume: int
    breakeven: float
    be_move_pct: float
    exp_move_pct: float
    unusual: bool
    flow_ratio: float       # حجم الكول / حجم البوت
    contract_trend: str = ""


def pick_expiry(expiries: list[str], today: date, cfg: dict) -> str | None:
    o = cfg["options"]
    best, best_d = None, 999
    for e in expiries:
        dte = (date.fromisoformat(e) - today).days
        if o["min_dte"] <= dte <= o["max_dte"] and abs(dte - 7) < best_d:
            best, best_d = e, abs(dte - 7)
    return best


def select_contract(S: float, calls: pd.DataFrame, puts: pd.DataFrame, expiry: str,
                    bias: str, hv: float, cfg: dict, today: date) -> Pick | None:
    o = cfg["options"]
    dte = (date.fromisoformat(expiry) - today).days
    T = max(dte, 0.5) / 365
    is_call = bias != "bear"
    chain = (calls if is_call else puts).copy()
    if chain.empty:
        return None

    # تذبذب ضمني عند السعر الحالي — للحركة المتوقعة
    def atm_iv(df):
        if df.empty:
            return np.nan
        row = df.iloc[(df["strike"] - S).abs().argsort()[:1]]
        return float(row["impliedVolatility"].iloc[0])
    ivs = [v for v in (atm_iv(calls), atm_iv(puts)) if v and not np.isnan(v) and v > 0.02]
    iv_atm = float(np.mean(ivs)) if ivs else hv
    exp_move = S * iv_atm * math.sqrt(T)

    c_vol = float(calls["volume"].fillna(0).sum()) if not calls.empty else 0.0
    p_vol = float(puts["volume"].fillna(0).sum()) if not puts.empty else 0.0
    flow = c_vol / p_vol if p_vol > 0 else (9.9 if c_vol else 1.0)

    for c in ("bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"):
        chain[c] = pd.to_numeric(chain[c], errors="coerce").fillna(0)
    chain["mid"] = np.where((chain["bid"] > 0) & (chain["ask"] > 0),
                            (chain["bid"] + chain["ask"]) / 2, chain["lastPrice"])
    chain["spread"] = np.where(chain["mid"] > 0, (chain["ask"] - chain["bid"]) / chain["mid"], 9)
    chain["iv"] = np.where(chain["impliedVolatility"] > 0.02, chain["impliedVolatility"], iv_atm)
    r = o["risk_free_rate"]
    chain["delta"] = [bs_delta(S, k, T, s, r, is_call) for k, s in zip(chain["strike"], chain["iv"])]
    chain["unusual"] = (chain["volume"] > 500) & (chain["volume"] > 1.5 * chain["openInterest"])

    f = chain[
        (chain["openInterest"] >= o["min_open_interest"]) &
        (chain["volume"] >= o["min_volume"]) &
        (chain["spread"] <= o["max_spread_pct"]) &
        (chain["mid"] >= o["min_premium"]) &
        (chain["delta"].abs().between(o["min_delta"], o["max_delta"]))
    ].copy()
    if f.empty:
        return None

    sign = 1 if is_call else -1
    f["be"] = f["strike"] + sign * f["mid"]
    f["be_move"] = sign * (f["be"] - S)
    f["score"] = (
        -abs(f["delta"].abs() - o["target_delta"]) * 100
        + np.log1p(f["openInterest"] + f["volume"]) * 3
        - f["spread"] * 50
        + np.where(f["be_move"] <= exp_move, 10, -10)
        + np.where(f["unusual"], 5, 0)
    )
    b = f.sort_values("score", ascending=False).iloc[0]
    return Pick(
        kind="CALL" if is_call else "PUT", contract=str(b["contractSymbol"]),
        strike=float(b["strike"]), expiry=expiry, dte=dte, mid=round(float(b["mid"]), 2),
        bid=float(b["bid"]), ask=float(b["ask"]), delta=round(float(b["delta"]), 2),
        iv=float(b["iv"]), oi=int(b["openInterest"]), volume=int(b["volume"]),
        breakeven=round(float(b["be"]), 2), be_move_pct=float(b["be_move"]) / S * 100,
        exp_move_pct=exp_move / S * 100, unusual=bool(chain["unusual"].any()), flow_ratio=round(flow, 2),
    )


def contract_trend(contract: str) -> str:
    """قراءة مبسطة لشارت العقد على فريم 15 دقيقة (آخر 5 أيام)."""
    try:
        h = yf.Ticker(contract).history(period="5d", interval="15m")
        if len(h) < 12:
            return "بيانات العقد غير كافية"
        c = h["Close"]
        ema = c.ewm(span=9).mean()
        vwap = (c * h["Volume"]).cumsum() / h["Volume"].replace(0, np.nan).cumsum()
        last = float(c.iloc[-1])
        up = last > float(ema.iloc[-1]) and last > float(vwap.iloc[-1])
        dn = last < float(ema.iloc[-1]) and last < float(vwap.iloc[-1])
        return "العقد فوق EMA9 وVWAP (زخم إيجابي)" if up else \
               "العقد تحت EMA9 وVWAP (زخم ضعيف)" if dn else "العقد متذبذب حول متوسطاته"
    except Exception:
        return ""


def options_for(setup: Setup, cfg: dict, today: date) -> Pick | None:
    try:
        t = yf.Ticker(setup.symbol)
        exp = pick_expiry(list(t.options), today, cfg)
        if not exp:
            return None
        ch = t.option_chain(exp)
        pick = select_contract(setup.price, ch.calls, ch.puts, exp, setup.bias, setup.hv, cfg, today)
        if pick:
            # نشاط العقود يدعم/يعاكس الاتجاه
            if pick.flow_ratio >= 2 and setup.bias == "bull" or pick.flow_ratio <= 0.5 and setup.bias == "bear":
                setup.score += 10; setup.reasons.append(f"تدفق عقود داعم (كول/بوت {pick.flow_ratio})")
            if pick.unusual:
                setup.score += 5; setup.reasons.append("نشاط عقود غير اعتيادي")
            pick.contract_trend = contract_trend(pick.contract)
        return pick
    except Exception as e:
        log.debug("عقود %s: %s", setup.symbol, e)
        return None


def earnings_soon(sym: str, days: int = 7) -> int | None:
    try:
        cal = yf.Ticker(sym).calendar
        d = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if d:
            d0 = d[0] if isinstance(d, list) else d
            delta = (pd.Timestamp(d0).date() - datetime.now(ET).date()).days
            if 0 <= delta <= days:
                return delta
    except Exception:
        pass
    return None


# ============================================================
# الرسائل
# ============================================================

def fmt_pick(p: Pick | None) -> list[str]:
    if not p:
        return ["العقد: لا يوجد عقد أسبوعي يحقق شروط السيولة"]
    lines = [
        f"العقد المرشح: {p.kind} {p.strike:g} — انتهاء {p.expiry} ({p.dte} يوم)",
        f"السعر ~{p.mid:.2f} (bid {p.bid:.2f} / ask {p.ask:.2f}) | دلتا {p.delta:+.2f} | IV {p.iv:.0%}",
        f"نقطة التعادل {p.breakeven:.2f} ({p.be_move_pct:+.1f}%) | الحركة المتوقعة ±{p.exp_move_pct:.1f}%",
        f"OI {p.oi:,} | حجم {p.volume:,} | {p.contract}",
    ]
    if p.contract_trend:
        lines.append(f"شارت العقد: {p.contract_trend}")
    return lines


def entry_message(s: Setup, p: Pick | None) -> str:
    up = s.bias == "bull"
    stop = s.up_trigger - 0.5 * s.atr if up else s.down_trigger + 0.5 * s.atr
    target = s.price * (1 + (p.exp_move_pct if p else s.hv * math.sqrt(7 / 365) * 100) / 100 * (1 if up else -1))
    head = "🚀 إشارة دخول — صعود" if up else "🔻 إشارة دخول — هبوط"
    lines = [
        head,
        f"{s.symbol}  ${s.price:.2f}  ({s.change_pct:+.1f}%)  | تقييم {s.score:.0f}",
        "السبب: " + " | ".join(s.reasons),
        *fmt_pick(p),
        f"هدف تقريبي للسهم {target:.2f} | إيقاف مقترح: إغلاق السهم {'تحت' if up else 'فوق'} {stop:.2f}",
        "⚠️ بيانات متأخرة ~15 دقيقة — للفرز وليست توصية",
    ]
    return "\n".join(lines)


def watchlist_message(items: list[tuple[Setup, Pick | None]]) -> str:
    now = datetime.now(RIYADH).strftime("%Y-%m-%d %H:%M")
    lines = [f"👀 أسهم مرشحة لحركة حادة — {now} (الرياض)", ""]
    for n, (s, p) in enumerate(items, 1):
        arrow = {"bull": "↑ ميول صعود", "bear": "↓ ميول هبوط"}.get(s.bias, "↕ محايد")
        lines.append(f"{n}) {s.symbol} ${s.price:.2f} ({s.change_pct:+.1f}%) — {arrow} — تقييم {s.score:.0f}")
        lines.append("   " + " | ".join(s.reasons[:4]))
        lines.append(f"   دخول: اختراق {s.up_trigger:.2f}↑ أو كسر {s.down_trigger:.2f}↓")
        if p:
            lines.append(f"   عقد: {p.kind} {p.strike:g} {p.expiry} ~{p.mid:.2f} (دلتا {p.delta:+.2f})")
        lines.append("")
    lines.append("⚠️ للفرز وليست توصية — راقب تنبيه الدخول")
    return "\n".join(lines)


def send_telegram(cfg: dict, text: str):
    tg = cfg["telegram"]
    if not tg.get("bot_token") or not tg.get("chat_id"):
        print("\n" + "=" * 60 + "\n" + text + "\n" + "=" * 60)
        return
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)]:
        try:
            r = requests.post(f"https://api.telegram.org/bot{tg['bot_token']}/sendMessage",
                              data={"chat_id": tg["chat_id"], "text": chunk,
                                    "disable_web_page_preview": True}, timeout=15)
            if not r.ok:
                log.error("تيليجرام رفض الرسالة: %s", r.text[:200])
        except Exception as e:
            log.error("فشل الإرسال لتيليجرام: %s", e)
    print(text)


# ============================================================
# دورة المسح
# ============================================================

def market_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    return now.replace(hour=9, minute=30, second=0) <= now <= now.replace(hour=16, minute=0, second=0)


def scan(cfg: dict, state: State, tickers: list[str]):
    now = datetime.now(ET)
    t0 = time.time()
    log.info("بدء المسح: %d سهم", len(tickers))
    data = download_daily(tickers)
    frac = session_fraction(now)

    setups = [s for sym, df in data.items() if (s := analyze_stock(sym, df, cfg, frac))]
    setups.sort(key=lambda s: (s.entry, s.score), reverse=True)
    top = [s for s in setups if s.score >= cfg["signals"]["min_setup_score"]][:cfg["signals"]["top_candidates_for_options"]]
    log.info("حلّلت %d سهم — %d مرشح قوي", len(setups), len(top))

    live = market_open(now)
    if live and setups:
        fresh = sum(s.bar_date == now.date() for s in setups) / len(setups)
        if fresh < 0.5:
            log.info("لا توجد شمعة لليوم (عطلة أو تأخر البيانات) — لا تنبيهات")
            return

    results: list[tuple[Setup, Pick | None]] = []
    for s in top:
        d = earnings_soon(s.symbol)
        if d is not None:
            s.score += 15; s.reasons.append(f"أرباح بعد {d} يوم" if d else "أرباح اليوم")
        results.append((s, options_for(s, cfg, now.date())))
    results.sort(key=lambda x: (x[0].entry, x[0].score), reverse=True)

    # 1) تنبيهات الدخول
    for s, p in results:
        if s.entry and s.bias != "neutral" and (not live or s.bar_date == now.date()):
            key = f"entry|{s.symbol}|{s.bias}"
            if not state.seen(key):
                send_telegram(cfg, entry_message(s, p))
                state.mark(key)

    # 2) قائمة المرشحين اليومية (مرة واحدة يومياً)
    if results and not state.seen("watchlist"):
        send_telegram(cfg, watchlist_message(results[:cfg["schedule"]["daily_watchlist_count"]]))
        state.mark("watchlist")

    rows = [{"symbol": s.symbol, "price": s.price, "chg%": round(s.change_pct, 2), "score": s.score,
             "bias": s.bias, "entry": s.entry, "rvol": round(s.rvol, 2), "up_trigger": s.up_trigger,
             "down_trigger": s.down_trigger, "reasons": " | ".join(s.reasons),
             "contract": p.contract if p else "", "strike": p.strike if p else "",
             "expiry": p.expiry if p else "", "mid": p.mid if p else "", "delta": p.delta if p else ""}
            for s, p in results]
    pd.DataFrame(rows).to_csv(BASE / "last_scan.csv", index=False, encoding="utf-8-sig")
    log.info("انتهى المسح في %.0f ثانية — النتائج في last_scan.csv", time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description="ماسح الأسهم والعقود الأسبوعية")
    ap.add_argument("--once", action="store_true", help="مسح واحد ثم خروج (يعمل حتى والسوق مغلق)")
    ap.add_argument("--test-telegram", action="store_true", help="إرسال رسالة اختبار")
    ap.add_argument("--tickers", nargs="+", help="مسح أسهم محددة بدل السوق كامل")
    ap.add_argument("--config", default=str(BASE / "config.json"))
    ap.add_argument("--market-hours-only", action="store_true",
                    help="مع --once: لا يمسح إذا السوق مغلق (للجدولة على GitHub)")
    ap.add_argument("--session", action="store_true",
                    help="جلسة يوم واحد: ينتظر الافتتاح، يمسح لين الإغلاق، ثم يقفل (للتشغيل التلقائي)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    cfg = load_config(Path(args.config))
    if args.test_telegram:
        send_telegram(cfg, "✅ الماسح متصل — ستصلك التنبيهات هنا.")
        return
    if yf is None:
        sys.exit("ثبّت المكتبات أولاً: pip install -r requirements.txt")

    state = State(BASE / "state.json")
    tickers = [t.upper() for t in args.tickers] if args.tickers else load_universe(cfg)

    if args.once:
        if args.market_hours_only and not market_open(datetime.now(ET)):
            log.info("السوق مغلق — لا مسح")
            return
        scan(cfg, state, tickers)
        return

    every = cfg["schedule"]["scan_every_minutes"] * 60
    if args.session:
        now = datetime.now(ET)
        close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now.weekday() >= 5 or now >= close_t:
            log.info("لا توجد جلسة تداول متبقية اليوم — خروج")
            return
    log.info("الماسح يعمل — يمسح كل %d دقيقة وقت السوق. Ctrl+C للإيقاف.", every // 60)
    while True:
        now = datetime.now(ET)
        if args.session and now >= now.replace(hour=16, minute=0, second=0, microsecond=0):
            log.info("أُغلق السوق — انتهت جلسة اليوم")
            send_telegram(cfg, "🔔 انتهت جلسة اليوم — الماسح توقف وسيعمل تلقائياً يوم التداول القادم.")
            return
        if market_open(now):
            state = State(BASE / "state.json")   # يتجدد مع كل يوم
            try:
                scan(cfg, state, tickers)
            except Exception as e:
                log.exception("خطأ في المسح: %s", e)
            time.sleep(every)
        else:
            log.info("السوق مغلق (%s بتوقيت الرياض) — انتظار…", datetime.now(RIYADH).strftime("%H:%M"))
            time.sleep(120 if args.session else 300)


if __name__ == "__main__":
    main()
