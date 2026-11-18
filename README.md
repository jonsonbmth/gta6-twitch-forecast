# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jonsonbmth/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jonsonbmth/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour forecast of Grand Theft Auto VI's launch-day Twitch viewers
(hours 2-24 of 2026-11-19 UTC), as five quantiles per hour. `main.py`
(Python 3.13, standard library only, no network) anchors to hour 1's print,
then takes a kernel-weighted mean of 38 past big launches' log-ratio curves
— weighted by pre-release ramp shape, ramp height, ramp peak vs hour 1 and
hour-1 scale, favouring rolling-midnight console launches — and calibrates
its bias and spread by leave-one-out cross-validation over the comparables
inside the forecast itself. The spread is therefore the forecaster's own
measured error on launches like the target's, not a guessed width.

Validation: practice pinball loss 0.177 (house baseline 0.387);
leave-one-out loss over the 37 comparable launches 0.409 (baseline 0.930);
leave-one-out on rolling-midnight console launches 0.136 (baseline 0.267).
Parameters were chosen on the LOO loss, which measures generalisation to
an unseen launch — the final case — rather than on the practice score.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Validate locally (format, determinism, runtime, LOO) with:

```sh
python3 evaluate.py <path-to-kit-input-dir>
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
