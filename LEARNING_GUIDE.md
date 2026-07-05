# 📚 Learning Guide — Understanding This Project From Zero

Welcome! This file is your **map**. You are new to coding, and this project has a lot of
files. Do **not** try to read everything at once. Read this guide top-to-bottom, and when
it tells you to open a Python file, open it and read the comments inside it (every file is
commented line-by-line), then come back here.

> **Golden rule:** Read the files in the *order this guide lists them*. That order is the
> same order the program actually runs. If you read them in a random order, nothing will
> make sense. If you read them in this order, each file only uses ideas you already saw.

---

## Part 0 — What does this project even do? (the 1-minute version)

We are predicting the **finishing order of Formula 1 races** — who comes 1st, 2nd, 3rd, …
all the way to last.

The program works like an assembly line with 4 stations. Data flows left to right:

```
   STATION 1            STATION 2            STATION 3           STATION 4
   ─────────            ─────────            ─────────           ─────────
  INGEST      ──►      FEATURES     ──►     TRAIN + TEST   ──►   PREDICT
 (download            (turn raw            (teach a model      (use the model
  race results         numbers into         from history        to rank the
  from the             useful clues)        + measure how       grid for a race)
  internet)                                 good it is)
```

Each station is a folder of Python files. You run a station with a command:

| You type… | Station | What happens |
|---|---|---|
| `uv run f1pred ingest` | 1. Ingest | Downloads race results → saves a data file |
| `uv run f1pred features` | 2. Features | Reads that file → builds "clues" → saves a new file |
| `uv run f1pred backtest` | 3. Test | Measures how good the model is vs. simple baselines |
| `uv run f1pred train` | 3. Train | Trains the final model → saves it |
| `uv run f1pred predict --season 2026 --round 8` | 4. Predict | Predicts one race |

---

## Part 1 — A few words you MUST know first

You'll see these everywhere. Learn them once here.

- **Python file / module** — a file ending in `.py`. It contains **functions** (reusable
  chunks of code) and sometimes **classes** (blueprints for objects).
- **Function** — a named recipe. You "call" it with inputs (arguments) and it gives back an
  output (the "return" value). Example: `load_config()` returns the settings.
- **Import** — one file borrowing code from another. `from f1pred.config import get_config`
  means "go into the config file and grab the `get_config` function."
- **DataFrame** — think of an **Excel spreadsheet inside the program**: rows and named
  columns. It comes from a library called **pandas** (imported as `pd`). Almost all our
  data is a DataFrame. One row = one driver in one race.
- **parquet** — a file format for saving a DataFrame to disk (like `.xlsx` but faster and
  smaller). Our saved data files end in `.parquet`.
- **library / package** — code other people wrote that we reuse. Ours: `pandas` (tables),
  `numpy` (fast math, imported as `np`), `fastf1` (F1 data), `lightgbm` (the model).
- **leakage** — the #1 danger in this kind of project. It means accidentally letting the
  model see the answer (the race result) while it's supposed to be *predicting* it. We work
  very hard to prevent this. A whole file (`leakage.py`) exists just for that.
- **feature** — a "clue" the model uses to predict. E.g. "this driver's average finish in
  the last 5 races." Turning raw results into features is called *feature engineering*.
- **model** — the thing that learns patterns from past races and then makes predictions.
  Ours is `LightGBM`.
- **learning-to-rank** — instead of predicting an exact number, the model learns to put
  drivers *in order*. Perfect for "who finishes ahead of whom."

Don't worry if these aren't fully clear yet — they'll click as you read the files.

---

## Part 2 — The folder map (where things live)

```
f1-pred-modal/
│
├── config.yaml          ← SETTINGS you can change (which seasons, model knobs). Plain text.
├── pyproject.toml       ← The project's "shopping list" of libraries + tool settings.
├── README.md            ← Project summary + results.
├── LEARNING_GUIDE.md    ← You are here.
│
├── src/f1pred/          ← ALL the real code lives here. "src" = source.
│   │
│   ├── config.py        ← Reads config.yaml into the program. (Foundation)
│   ├── logging_utils.py ← Helper for printing nice status messages. (Foundation)
│   ├── schema.py        ← The agreed list of column names. (Foundation)
│   ├── cli.py           ← The "front desk": turns your typed command into an action.
│   │
│   ├── ingest/          ← STATION 1: download data
│   │   ├── cache.py         ← tells FastF1 where to save downloads
│   │   ├── fastf1_loader.py ← downloads ONE race, cleans it into our columns
│   │   └── run.py           ← loops over many races, saves the big data file
│   │
│   ├── features/        ← STATION 2: build clues (features)
│   │   ├── leakage.py       ← safety tools that prevent "seeing the future"
│   │   ├── driver.py        ← clues about each driver's recent form
│   │   ├── constructor.py   ← clues about each team's car (pace + reliability)
│   │   ├── track.py         ← clues about how they did at this track before
│   │   ├── regs2026.py      ← special 2026 rule-change flags
│   │   └── build.py         ← runs all the above and saves the feature file
│   │
│   ├── models/          ← STATION 3 (part A): the models
│   │   ├── baselines.py     ← 3 "dumb" models to beat (proves our model is worth it)
│   │   └── ranker.py        ← the real LightGBM ranking model
│   │
│   ├── eval/            ← STATION 3 (part B): measuring quality
│   │   ├── metrics.py       ← the "scorecards" (NDCG, top-1, etc.)
│   │   └── backtest.py      ← the fair test: train on the past, predict the future
│   │
│   └── predict/         ← STATION 4: make one prediction
│       └── infer.py         ← predict a single race's finishing order
│
├── tests/               ← automatic checks that the code is correct
│   ├── test_config.py, test_metrics.py, test_leakage.py,
│   ├── test_ingest.py, test_features.py, test_models.py, test_cli.py
│
├── data/                ← where downloaded + built data files are saved (auto-created)
│   ├── raw/                 ← Station 1's output (results.parquet)
│   ├── features/            ← Station 2's output (features.parquet)
│   ├── predictions/         ← predictions + backtest scores
│   └── reference/entries_2026.csv ← hand-typed 2026 team/engine info
│
└── models/              ← the trained model gets saved here (ranker.pkl)
```

---

## Part 3 — THE READING ORDER (do this in sequence)

Below is the exact order to read the files. Each step says **why** you're reading it and
**what to notice**. Open the `.py` file, read its line-by-line comments, then return here
and go to the next step.

### 🧱 Group A: The Foundation (read these first — everything depends on them)

**Step 1 — `src/f1pred/config.py`**
The project's settings desk. Every other file starts by asking it "what are the settings?"
via `get_config()`. It reads `config.yaml` (open that too — it's just plain text you can
edit). *Notice:* the idea that settings live in ONE place so you never hard-code values.

**Step 2 — `src/f1pred/logging_utils.py`**
Tiny file. Gives every other file a `log` object so they can print tidy progress messages
like `12:00:05 INFO ... loaded 2026 R8`. *Notice:* why we use `log.info(...)` instead of
`print(...)` (timestamps + the file's name are added automatically).

**Step 3 — `src/f1pred/schema.py`**
Just a list of the column names our data table will have (`season`, `round`, `driver_id`,
`position`, …). It's the "contract" all files agree on. *Notice:* `TARGET_COLUMN` (the thing
we predict = `position`) and `GROUP_KEY` (what defines one race).

### 🏎️ Group B: Station 1 — Ingestion (getting the data)

**Step 4 — `src/f1pred/ingest/cache.py`**
Before downloading, we tell FastF1 to save everything in a folder so we never download the
same race twice. *Notice:* "idempotent" = calling it twice is harmless.

**Step 5 — `src/f1pred/ingest/fastf1_loader.py`**
Downloads **one** race and *translates* FastF1's columns into OUR column names (from
`schema.py`). *Notice:* it's split into two parts — `to_canonical` (pure translation, no
internet) and `load_session_results` (the part that hits the internet). This split lets us
test the translation without needing a connection.

**Step 6 — `src/f1pred/ingest/run.py`**
The loop. It goes season by season, round by round, calls Step 5 for each race, and saves
one big `results.parquet`. *Notice:* it *skips* races it already has (so re-running is fast)
and *skips* races that haven't happened yet.

👉 After this group, run `uv run f1pred ingest` and look at `data/raw/results.parquet`
exists. You just did Station 1!

### 🛠️ Group C: Station 2 — Features (the heart of the project)

**Step 7 — `src/f1pred/features/leakage.py`** ⭐ *most important concept in the project*
The safety tools. Two functions, `shifted_rolling` and `expanding_prior`, both compute
"averages of PAST races only." The trick is `.shift(1)` which drops the current race so the
model can never peek at the result it's trying to predict. Read this slowly.

**Step 8 — `src/f1pred/features/driver.py`**
Uses Step 7 to build driver clues: average finishing position over the last 3/5/10 races,
how often they beat their teammate, career totals, etc.

**Step 9 — `src/f1pred/features/constructor.py`**
Same idea but for the *team's car*: recent points, best finish, reliability (how often the
car breaks). Built at the team level then copied onto both drivers.

**Step 10 — `src/f1pred/features/track.py`**
Clues about this specific circuit: how the driver/team has historically done *here*.

**Step 11 — `src/f1pred/features/regs2026.py`**
Special flags for the 2026 rule reset (new engines, new team Cadillac, etc.), read from the
hand-typed CSV in `data/reference/`.

**Step 12 — `src/f1pred/features/build.py`**
The conductor. It runs Steps 8–11 in order, adds the cleaned `grid_start` clue, decides
which columns are "features" vs. "labels/ids", and saves `features.parquet`.

👉 Run `uv run f1pred features`. You just did Station 2!

### 📏 Group D: Station 3 — Models & Measuring

**Step 13 — `src/f1pred/eval/metrics.py`**
The scorecards. Given the model's predicted order and the real result, these functions
output numbers like "did we get the winner right?" (`top1_hit`) and NDCG (a ranking score).
*Notice:* no model here yet — just math that grades a prediction.

**Step 14 — `src/f1pred/models/baselines.py`**
Three simple, "dumb" predictors: (1) *grid* = predict they finish where they started,
(2) *driver-form* = predict by recent average finish, (3) *constructor-Elo* = a team
strength rating. We must beat these to prove our real model is worth anything.

**Step 15 — `src/f1pred/models/ranker.py`**
The real model: a `LightGBM` learning-to-rank model. Wraps it in a friendly class with
`.fit()` (learn) and `.predict()` (guess) and `.save()`/`.load()`.

**Step 16 — `src/f1pred/eval/backtest.py`** ⭐ *the "is it actually good?" file*
The fair exam. It trains the model on old seasons and tests it on newer, *unseen* seasons —
walking forward through time, never letting the model see the future. Then it prints the
comparison table (our ranker vs. the 3 baselines). This is what produced the results in the
README.

👉 Run `uv run f1pred backtest` and read the printed table.

### 🔮 Group E: Station 4 — Predicting one race

**Step 17 — `src/f1pred/predict/infer.py`**
Takes a season + round, trains on everything *before* that race, and prints the predicted
finishing order next to the real one. This is what a demo/web app would call.

👉 Run `uv run f1pred predict --season 2026 --round 8`.

### 🚪 Group F: The front desk (how a command becomes an action)

**Step 18 — `src/f1pred/cli.py`**
When you type `f1pred ingest`, THIS file catches the word "ingest" and calls
`run_ingest()`. It's the switchboard connecting your keyboard to the stations. Read it last,
now that you know all the stations it points to.

### ✅ Group G: The tests (optional but great for learning)

**Step 19 — the `tests/` folder**
Each `test_*.py` file feeds tiny fake data into one part of the code and checks the answer
is correct. `tests/test_leakage.py` and `tests/test_features.py` are the most instructive —
they *prove* the no-peeking-at-the-future guarantee. Reading tests is a great way to see how
each function is *meant* to be used.

---

## Part 4 — The whole story in one paragraph (read after the files)

You type `f1pred ingest`. **cli.py** catches it and calls **ingest/run.py**, which uses
**cache.py** and **fastf1_loader.py** to download every race and save `results.parquet` with
the columns defined in **schema.py**, using settings from **config.py**. Next you type
`f1pred features`; **features/build.py** loads that file and runs **driver.py**,
**constructor.py**, **track.py**, and **regs2026.py** — all built on the anti-cheating tools
in **leakage.py** — to save `features.parquet`. Then `f1pred backtest` (**eval/backtest.py**)
repeatedly trains the model from **models/ranker.py**, compares it to **models/baselines.py**
using the scorecards in **eval/metrics.py**, and prints how well we did. `f1pred train` saves
the final model, and `f1pred predict` (**predict/infer.py**) uses it to rank one race. Every
file logs its progress through **logging_utils.py**. That's the entire project. 🏁

---

## Part 5 — How to actually run it (cheat sheet)

Open a terminal in the project folder and type:

```bash
uv sync --extra dev                 # 1. install all libraries (only needed once)
uv run pytest -m "not network"      # 2. run the automatic checks (should say "passed")
uv run f1pred ingest                # 3. download data  (Station 1)
uv run f1pred features              # 4. build features (Station 2)
uv run f1pred backtest              # 5. measure quality (Station 3)
uv run f1pred train                 # 6. train + save the final model
uv run f1pred predict --season 2026 --round 8   # 7. predict one race (Station 4)
```

`uv run` just means "run this using the project's installed libraries." If a command
errors saying a file is missing, it usually means you skipped an earlier step (e.g. you must
`ingest` before you can build `features`).

---

## Part 6 — Tips for reading code as a beginner

1. **Read the docstring first.** The `"""triple-quoted text"""` at the top of each file and
   function explains its purpose in English. Read that before the code.
2. **Ignore `from __future__ import annotations`** — it's a technical line at the top of
   every file that improves type hints. It does nothing you need to think about.
3. **Follow the data, not every line.** Ask "what shape is the data before this line, and
   after?" more than "what does every symbol mean?"
4. **`_underscore` names are "internal helpers."** A function like `_beat_teammate` is a
   private helper used only inside its own file. Public functions have no leading underscore.
5. **Run it and print things.** The best way to learn is to add a `print(df.head())` after a
   line and re-run, to *see* the data at that point.
6. **You don't need to understand LightGBM's internals** to understand this project. Treat
   the model as a box: you `.fit()` it on past races and `.predict()` new ones.

Happy learning — go to **Step 1** now. 🚦
