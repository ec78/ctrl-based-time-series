# Let the LLM Be the Coach, Not the Player: A Look at CTRL for Time Series Forecasting

*A plain-English tour of "CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning" (Kim et al., 2026)*

> **A note on evidence.** All benchmark numbers quoted below are the authors' own, reported in the paper. I have not independently reproduced them: the authors' linked code repository currently contains only a placeholder README. The toy example in section 4 is my own simplified illustration, not a replication.

---

## The short version

Large language models (LLMs) are good at *reasoning*. They are not great at *arithmetic*. Most attempts to use them for forecasting ask them to do the arithmetic anyway: read in numbers, spit out numbers.

CTRL takes a different route. A conventional forecasting model still produces the numbers. The LLM plays **coach**: it studies how the forecasting model has been going wrong and issues a short list of instructions, such as "you're flattening the daily cycle, push it out by about 40%." A very small model then turns those instructions into a correction.

The result is a forecast that adapts better when the world changes, at a fraction of the cost of other LLM-based approaches.

---

## 1. LLMs meet time series

Language models have shown up in forecasting in a few flavors:

- **"Forecasting as text completion."** Write the numbers out as text (`"12.1, 12.4, 12.9, ..."`) and ask the model to continue the sequence. (LLMTime, PromptCast.)
- **"Repurposing the LLM's machinery."** Translate the series into a form a pretrained language model understands, then lightly fine-tune it. (Time-LLM, GPT4TS.)
- **"Aligning numbers with language."** Build extra structure so time-series patterns and text-like representations line up. Often this includes splitting a series into trend, seasonality, and leftover noise. (TEMPO, CALF, TimeCMA.)

The appeal is intuitive. LLMs have absorbed a huge amount of knowledge about how the world works, and they can reason about *why* something might be happening. Maybe that helps forecasting.

## 2. Where LLM-based forecasters fall short

The paper's argument is that most of these approaches share two problems.

**They use the LLM for the thing it is worst at.** Predicting the next number precisely is a job that purpose-built forecasting models already do very well. Several recent studies the paper cites find that the LLM's real value lies in reasoning about the series, not in producing exact values, and that LLM forecasts can be surprisingly fragile. Small amounts of noise can throw them off.

**They tie the LLM directly to the forecast.** When the LLM's output *is* the prediction, any wobble in the LLM shows up in the forecast. This is most damaging when the data is **non-stationary**, meaning the underlying behavior is shifting: a new trend, a change in seasonal swing, a regime change.

There are practical costs too:

| Approach | Practical burden |
|---|---|
| Prompt-only (e.g. LLMTime) | One LLM call *per forecast*. Expensive at scale. |
| Fine-tuned alignment methods (CALF, TEMPO) | 12–18 million trainable parameters and 1–2 hours of training in the paper's setup. |
| Alignment in general | Tend to assume the future looks like the training data. |

**How does this compare with the alternatives?** A word of context, since the paper's experiments don't cover everything:

- **Classical statistical models** (ARIMA, exponential smoothing, and friends) are transparent, cheap, and hard to beat on well-behaved, stable series. They can't use outside knowledge, and they need re-fitting when behavior changes.
- **Time series foundation models** (TSFMs, such as Google's TimesFM) are pretrained on huge collections of series and forecast "zero-shot." They are strong numerical forecasters, but they are large and they don't reason about *why* they're missing.
- **Lean deep-learning models** (DLinear, PatchTST) are fast, accurate, and are what CTRL itself is built on top of.

To be upfront: the paper benchmarks against the deep-learning and LLM-based models above. It does **not** test classical statistical models, and it explicitly leaves TSFMs like TimesFM for future work. So the comparison in this section is the general landscape, not something the paper measured.

## 3. Meet CTRL

The name stands for **C**ontrol-based **T**ime series forecasting with LLM-guided **R**esidual **L**earning. The core idea fits in one line:

> **Final forecast = backbone forecast + a learned correction, where an LLM decides what kind of correction is needed.**

Here is the cast of characters.

**The backbone (frozen).** Any ordinary forecasting model, trained once and then left alone. The paper uses DLinear and PatchTST. This is the "player." It does the numbers.

**The controllers (LLM agents).** Three LLM "specialists," one for each ingredient of a time series:

- a **Trend agent** (is the backbone getting the direction and slope right?),
- a **Seasonal agent** (is it flattening the daily or weekly cycle?),
- an **Irregular agent** (how noisy or surprising is what's left over?).

They look at the backbone's *past mistakes* on a set of held-out data. Importantly, they never see the raw series, only summaries such as "the backbone underestimates trend slope by 15%." The paper argues this also sidesteps the worry that an LLM has memorized the benchmark data.

**The control signals.** Each numerical agent hands back four numbers:

| Signal | Plain-English meaning |
|---|---|
| **scale** | Should this component be turned up or down? |
| **bias** | Is the forecast systematically too high or too low? |
| **gate** | How aggressively should the correction be applied? |
| **confidence** | How sure is the agent? |

The Irregular agent writes a short natural-language note instead, which gets converted into numbers by a frozen text encoder.

**The residual decoder.** A tiny neural network (about 400,000 parameters) that reads the control signals and produces the actual numerical correction. It is the *only* part that gets trained, and it trains in under two minutes.

**Test-time adaptation.** This is the clever part for a changing world. As new data comes in, CTRL compares simple statistics of the incoming inputs (for example, how strong the seasonal swing looks) against what it saw during setup. If something has moved by more than about two standard deviations, the LLM is asked again and the control signals are adjusted. No true future values are needed for this, which matters because in real deployments the answers arrive late or never. Because the signals are cached and shared, the whole system needs only **3 to 24 LLM calls in total**, not one per forecast.

```
   history ──► frozen backbone ──► base forecast ──────────┐
      │                                                    ▼
      │   (past backbone mistakes)                    ┌─────────┐
      └──► LLM controllers ──► control signals ─────► │ residual│──► + ──► final forecast
            Trend / Seasonal / Irregular              │ decoder │
                  ▲                                   └─────────┘
                  └── re-consulted only if input statistics shift
```

### What the paper found

- **Best gains where data shifts.** On datasets with the biggest change between training and test (the ETTh2 and ETTm2 electricity-transformer datasets), CTRL cut error (MSE) by up to about **12%** relative to its backbone.
- **Competitive elsewhere.** On stable data, gains are small, and on the Weather dataset the fine-tuned CALF model actually did better at some horizons. The authors describe this as expected: with little shift, there is little to adapt to.
- **Beats "LLM as predictor."** Against LLMTime (GPT-4 forecasting directly), CTRL's MSE was **19.6% to 44.3% lower** on the two datasets compared.
- **Much lighter.** About 0.4M trainable parameters, versus 12.4M for TEMPO and 18.2M for CALF, and under 2 minutes of training versus roughly 1 to 2 hours.
- **The LLM's words matter.** When the authors swapped in random numbers or gibberish text for the real control signals, the gains disappeared. This suggests the improvement isn't just extra model capacity.

### Caveats from the authors

Improvements shrink at long forecast horizons, where the backbone's errors are larger and corrections are less certain. And CTRL has not yet been tried as a correction layer on top of foundation models like TimesFM.

## 4. A toy example

To make this concrete, I built a small simulation that mimics CTRL's flow. The code is in [`toy_ctrl.py`](toy_ctrl.py) and runs in a few seconds.

> **Important honesty note.** I did not call a real LLM. The "controller" in this toy is a simple rule-based stand-in that reads the same kind of summary an LLM agent would and outputs the same four signals. The script prints the text an LLM would have been shown. So this toy illustrates the *plumbing and the logic* of CTRL. It is not a reproduction of the paper's results, and the numbers below say nothing about how a real LLM would perform.

**The setup.** An hourly series with a steady upward trend and a daily cycle. Everything looks the same through the training and held-out periods. Then, in the test period, there is a regime change: **the daily swing gets about 60% bigger** while the trend stays put.

![The toy series: held-out versus test period](toy_series.png)

**The backbone.** A frozen model that learned the average daily shape from training data, but, like many smooth models, slightly flattens it and damps the trend.

**Step 1: the controller reviews the mistakes.** Looking at the held-out period, the controller sees the backbone is under-predicting the seasonal swing (predicted amplitude roughly 2.4 versus an actual of about 3.5, in the toy's units). It issues a signal along the lines of *seasonal: scale up ×1.44, gate 0.6, confidence 0.8.*

**Step 2: the decoder learns to apply it.** The tiny decoder learns how much weight to give those instructions, using the held-out data.

**Step 3: the world changes.** At test time, CTRL checks the input statistics. Result: the **trend shift score is about 0.0** (nothing changed), while the **seasonal shift score is about 19** (far above the threshold of 2). So only the seasonal signal is reconsidered: scale is pushed to its maximum, more authority is handed to the decoder, and confidence is lowered. The trend correction is left alone. This mirrors the "component-by-component" adaptation in the paper.

**The results on the test period:**

![Forecast error after the regime change](toy_mse.png)

| Model | Test MSE | Change vs. backbone |
|---|---|---|
| Backbone only | 1.39 | n/a |
| CTRL, no adaptation | 0.97 | −30% |
| CTRL, full (with adaptation) | 0.94 | −33% |

And here is what it looks like on a single 24-hour forecast:

![One forecast window after the regime change](toy_forecast.png)

The backbone (red dashed) draws too gentle a daily cycle. The CTRL versions (orange and blue) stretch it toward the truth (black). They still fall short of the real peak, because the controller's scale is capped at 1.5, which is a design choice in the paper as well.

**What to take from the toy:**

1. **The two-part structure works.** Leaving the numbers to a stable model and using a separate "reasoner" to spot systematic mistakes gives a big, cheap improvement.
2. **Shift detection is targeted.** It flagged exactly the component that changed and left the other alone.
3. **Adaptation was a modest extra.** In this toy, most of the gain came from the initial control signals, and adaptation added about three more points. That matches the paper's own finding, where adaptation's average contribution is small (+0.7%) but larger on the hardest samples.
4. **Limits are visible too.** When the shift is bigger than the controller is allowed to correct, some error remains. This is one run of one simulated series, so treat the exact percentages as illustrative.

If you would like to go one step further, the stand-in `controller()` and `adapt_controller()` functions in the script are the natural places to swap in real LLM calls.

## 5. Takeaways

- LLMs are strongest as **reasoners**, not calculators. CTRL is built around that observation.
- Splitting the job (a stable model for numbers, an LLM for diagnosis, a tiny decoder for the correction) makes the system **cheap, interpretable, and robust** to change.
- It is most useful when **your data drifts** and labels arrive late. It offers little when your data is very stable.
- Open questions remain, including how it behaves on top of foundation models and how it compares with classical statistical baselines, since the paper doesn't test either.

---

**Reference:** Kim, M., Ji, D., Lee, Y., Kim, B., & Jang, B. (2026). *CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning.* arXiv:2609.23257. Code repository (announced by the authors, but as of late September 2026 it contains only a placeholder README and no code): https://github.com/mfriendly/CTRL
