"""
app.py — Flask server for NQ HVN heatmap viewer

Run:
    python app.py
Then open http://localhost:5000
"""

import io
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, Response

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vol_profile as vp
import chart as ch

app = Flask(__name__, template_folder="templates")
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(BASE_DIR, "cache", "profiles")

# ── heatmap cache keyed by (date, session, lookback, smooth, hvn_pct) ────────
_heat_cache: dict = {}

def _get_heatmap(date_nodash: str, session: str, lookback: int,
                 smooth: int, hvn_pct: float):
    key = (date_nodash, session, lookback, smooth, hvn_pct)
    if key not in _heat_cache:
        _heat_cache[key] = ch.build_shelf_confluence_heatmap(
            date_nodash, session=session, lookback=lookback, smooth=smooth)
    return _heat_cache[key]


# ── model predictions (Stage-3 winner: nearest_hvn lb=15 ms=6 day+week+month) ─
_pred_cache: dict | None = None

def _get_predictions() -> dict:
    """date_nodash → (pred_top, pred_bot) from best Stage-3 combo."""
    global _pred_cache
    if _pred_cache is not None:
        return _pred_cache
    _pred_cache = {}
    fp = os.path.join(BASE_DIR, "results", "stage3_results.csv")
    if os.path.exists(fp):
        import pandas as pd
        df = pd.read_csv(fp, dtype={"date": str})
        best = df[
            (df.strategy == "nearest_hvn") & (df.lookback == 15) &
            (df.min_stack == 6) & (df.anchored == "day+month+week") &
            (df.pct == 25.0) & (df.depth_pct == 60.0) & (df.smooth == 5)
        ]
        for _, r in best.iterrows():
            pt = r["pred_top"] if pd.notna(r["pred_top"]) else None
            pb = r["pred_bot"] if pd.notna(r["pred_bot"]) else None
            _pred_cache[str(r["date"])] = (pt, pb)
    return _pred_cache


def _prior_dates(date_nodash: str, lookback: int, session: str) -> list[str]:
    all_dates = sorted(
        f.replace(f"_{session}.npy", "")
        for f in os.listdir(PROFILE_DIR)
        if f.endswith(f"_{session}.npy")
    )
    return [d for d in all_dates if d < date_nodash][-lookback:]


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    # list all cached session dates (full)
    try:
        dates = sorted(set(
            f.replace("_full.npy", "")
            for f in os.listdir(PROFILE_DIR)
            if f.endswith("_full.npy")
        ), reverse=True)
    except FileNotFoundError:
        dates = []
    default_date = dates[len(dates) // 2] if dates else ""
    return render_template("index.html", dates=dates, default_date=default_date)


@app.route("/chart")
def chart_endpoint():
    date_str   = request.args.get("date", "")
    start_t    = request.args.get("start",    "09:30")
    end_t      = request.args.get("end",      "16:00")
    lookback   = int(request.args.get("lookback",  15))
    session    = request.args.get("session",  "full")
    hvn_pct    = float(request.args.get("hvn_pct", 60.0))
    smooth     = int(request.args.get("smooth", 5))
    weighting  = request.args.get("weighting", "recency")
    pred_high  = request.args.get("pred_high", None)
    pred_low   = request.args.get("pred_low",  None)

    if pred_high is not None:
        try: pred_high = float(pred_high)
        except ValueError: pred_high = None
    if pred_low is not None:
        try: pred_low = float(pred_low)
        except ValueError: pred_low = None

    if not date_str:
        return Response("date required", status=400)

    date_nodash = date_str.replace("-", "")

    # auto-fill predictions from Stage-3 best model if not supplied
    if pred_high is None and pred_low is None:
        preds = _get_predictions().get(date_nodash)
        if preds:
            pred_high, pred_low = preds

    try:
        import numpy as np
        import pandas as pd
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")

        price_grid = ch._load_price_grid()
        prior_dates = _prior_dates(date_nodash, lookback, session)

        if not prior_dates:
            return Response(f"No prior sessions before {date_str}", status=400)

        # ── OHLCV ──
        ohlcv = None
        cached_ohlcv = ch._load_cached_ohlcv(date_nodash)
        if cached_ohlcv is not None:
            sess_key = "rth" if start_t >= "09:30" and end_t <= "16:01" else "full"
            sub = cached_ohlcv[cached_ohlcv["session"] == sess_key]
            if not sub.empty:
                ohlcv = sub.drop(columns=["session"])

        if ohlcv is None or ohlcv.empty:
            trades = ch._load_cached_trades(date_nodash)
            if trades is None:
                return Response(f"No data for {date_str}", status=404)
            trades_et = trades.tz_convert(ET)
            s_et = pd.Timestamp(f"{date_str} {start_t}", tz=ET)
            e_et = pd.Timestamp(f"{date_str} {end_t}",   tz=ET)
            window = trades_et[(trades_et.index >= s_et) & (trades_et.index <= e_et)]
            ohlcv  = ch._resample_ohlcv(window)

        if ohlcv is None or ohlcv.empty:
            return Response(f"No trades in {date_str} {start_t}–{end_t}", status=400)

        # apply time window if full-session OHLCV returned
        if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
            ohlcv_et = ohlcv.copy()
            ohlcv_et.index = ohlcv.index.tz_convert(ET)
            s_et = pd.Timestamp(f"{date_str} {start_t}", tz=ET)
            e_et = pd.Timestamp(f"{date_str} {end_t}",   tz=ET)
            ohlcv_et = ohlcv_et[(ohlcv_et.index >= s_et) & (ohlcv_et.index <= e_et)]
            ohlcv    = ohlcv_et.copy()
            ohlcv.index = ohlcv.index.tz_convert("UTC")

        if ohlcv.empty:
            return Response(f"No bars in {date_str} {start_t}–{end_t}", status=400)

        price_lo = float(ohlcv["low"].min())  - 30
        price_hi = float(ohlcv["high"].max()) + 30

        # ── HVN confluence heatmap ──
        heat = _get_heatmap(date_nodash, session, lookback, smooth, hvn_pct)

        # ── draw ──
        x_nums = np.arange(len(ohlcv))

        fig, (ax_main, ax_vol) = plt.subplots(
            2, 1, figsize=(18, 11),
            gridspec_kw={"height_ratios": [5, 1]},
            facecolor=ch.BG_COLOR,
        )
        fig.subplots_adjust(hspace=0.0, left=0.06, right=0.97, top=0.93, bottom=0.07)

        for ax in [ax_main, ax_vol]:
            ax.set_facecolor(ch.BG_COLOR)
            ax.tick_params(colors=ch.TEXT_COLOR, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#111111")
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

        if heat is not None:
            pg_full, intensity = heat
            ch.draw_heatmap(ax_main, pg_full, intensity,
                            x_lo=-1, x_hi=len(ohlcv),
                            price_lo=price_lo, price_hi=price_hi)
        ch.draw_candles(ax_main, ohlcv, x_nums, alpha=0.65)

        if pred_high is not None:
            ax_main.axhline(pred_high, color=ch.PRED_HIGH_COLOR,
                            linewidth=1.5, linestyle="--", zorder=5)
            ax_main.annotate(f"Pred High  {pred_high:.2f}",
                             xy=(len(ohlcv)-1, pred_high), xytext=(4,2),
                             textcoords="offset points", color=ch.PRED_HIGH_COLOR,
                             fontsize=8, annotation_clip=False)

        if pred_low is not None:
            ax_main.axhline(pred_low, color=ch.PRED_LOW_COLOR,
                            linewidth=1.5, linestyle="--", zorder=5)
            ax_main.annotate(f"Pred Low   {pred_low:.2f}",
                             xy=(len(ohlcv)-1, pred_low), xytext=(4,2),
                             textcoords="offset points", color=ch.PRED_LOW_COLOR,
                             fontsize=8, annotation_clip=False)

        ax_main.set_xlim(-1, len(ohlcv))
        ax_main.set_ylim(price_lo, price_hi)
        ax_main.yaxis.tick_right()
        ax_main.yaxis.set_tick_params(labelcolor=ch.TEXT_COLOR)

        ohlcv_et2 = ohlcv.copy()
        if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
            ohlcv_et2.index = ohlcv.index.tz_convert(ET)
        ch._format_x_ticks(ax_main, ohlcv_et2)
        ax_main.set_xticklabels([])

        vol_colors = ["#444444" if r["close"] >= r["open"] else "#333333"
                      for _, r in ohlcv.iterrows()]
        ax_vol.bar(x_nums, ohlcv["volume"], color=vol_colors, width=0.8, zorder=2)
        ax_vol.set_xlim(-1, len(ohlcv))
        ax_vol.set_ylim(0, ohlcv["volume"].max() * 1.3)
        ch._format_x_ticks(ax_vol, ohlcv_et2)
        ax_vol.yaxis.tick_right()
        import matplotlib.ticker
        ax_vol.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            )
        )

        ax_main.set_title(
            f"NQ  {date_str}  {start_t}–{end_t} ET  |  "
            f"{len(ohlcv)} bars  |  HVN confluence heatmap  "
            f"(lb={lookback}d day+week+month recency sess={session})",
            color=ch.TEXT_COLOR, fontsize=9, pad=6,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor=ch.BG_COLOR)
        plt.close(fig)
        buf.seek(0)

        meta = json.dumps({
            "date": date_str, "start": start_t, "end": end_t,
            "bars": len(ohlcv),
            "pred_high": pred_high, "pred_low": pred_low,
            "lookback": lookback, "session": session,
        })

        return Response(buf.read(), mimetype="image/png",
                        headers={"X-Chart-Meta": meta})

    except Exception as exc:
        app.logger.exception("Chart generation failed")
        return Response(str(exc), status=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  NQ HVN Heatmap → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
