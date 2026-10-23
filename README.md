## Installation

Requires Python 3.10–3.12.

With [uv](https://docs.astral.sh/uv/) (installs the right Python itself):

```
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

Or with pip:

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Running the experiments

All runs from the paper (2 environments × 4 methods × 3 seeds), per noise level:

```
python dispatch.py 7 0.0
python dispatch.py 7 0.1
python dispatch.py 7 0.2
```

The first number is how many runs execute at the same time. Logs go to `results/`, models to `models/`.
After a crash, run the same command again; unfinished runs continue from their checkpoint.
Expect long run times: SDMPS and SMPS are slow because of the planning and the statistical check.

A single run:

```
python main.py --env DoubleGatePoint --mode sdmps --noise 0.1 --seed 0 --log_to results/test.log
```

- `--env`: `DoubleGatePoint` or `ThickDoubleGatePoint`
- `--mode`: `td3`, `dmps`, `smps` or `sdmps`
- `--noise`: standard deviation of the actuator noise (0.0, 0.1 or 0.2)
- For DMPS under noise, the paper uses the hold backup: add `--hold`.

## Results

The logs behind the paper are in `results/noise0.0`, `results/noise0.1` and `results/noise0.2`.
`analyse_resultaten.ipynb` reads them and builds the results table and figures.
