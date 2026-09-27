# T_sLSTM: Transformer + sLSTM on P-sLSTM

P-sLSTM with a Transformer encoder between the patch embedding and the sLSTM
stack. Everything else — patching, channel independence, the sLSTM blocks, the
flatten head, and every hyperparameter in the run scripts — is P-sLSTM's own, so
a T_sLSTM vs P_sLSTM comparison isolates what the encoder adds.

```
input [B, L, C]
 -> channel independence + patching         [B*C, P, patch_size]
 -> Linear embedding                        [B*C, P, d]
 -> + sinusoidal positions                  (new)
 -> TransformerEncoder, t_layers x t_heads  (new)  global context across patches
 -> sLSTM stack                             [B*C, P, d]   local dynamics, in order
 -> flatten -> Linear head                  [B, pred_len, C]
```

The encoder is PyTorch's `nn.TransformerEncoder` (post-norm, feed-forward 4d,
dropout from `--dropout`), the same arrangement as the AGAP backbone.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--t_layers` | 2 | encoder layers; `0` skips the encoder and reproduces P_sLSTM exactly |
| `--t_heads` | 4 | attention heads; must divide `--embedding_dim` |
| `--backend` | `vanilla` | sLSTM kernel: `vanilla` (pure PyTorch, CPU or GPU) or `cuda` (compiled; needs the CUDA toolkit and ninja). Applies to P_sLSTM too |

## Setup (Windows, CPU)

```bash
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-t_slstm.txt
mkdir -p dataset
curl -L -o dataset/weather.csv https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv
```

`requirements-t_slstm.txt` pins every package to the exact versions this was
built and tested with, CPU-only PyTorch included, so both collaborators get the
same environment. Use it instead of the upstream `requirements.txt`, which pins
numpy 1.26; this code runs on numpy 2 after one fix (`np.Inf` -> `np.inf` in
`utils/tools.py`).

## Run

```bash
.venv/Scripts/python -m pytest tests -q                  # model checks
source .venv/Scripts/activate
bash scripts/EXP-LongForecasting/T_sLSTM/weather.sh      # all four horizons
bash scripts/EXP-LongForecasting/P_sLSTM/weather.sh      # baseline, same settings
```

Logs go to `logs/LongForecasting/`, metrics are appended to `result.txt`. Each
test run also writes the full prediction arrays to `results/<setting>/`
(~440 MB for Weather at horizon 96); delete them between runs if disk matters.

`.gitattributes` keeps the `.sh` scripts LF on every checkout, even with
`core.autocrlf=true`; CRLF line endings break bash line continuations.

## Measured cost

Weather, horizon 96, CPU (`vanilla`), batch 16, 2278 iterations per epoch:
504K parameters (P_sLSTM 262K). About 0.38 s per iteration on mains power, so
~15 minutes per epoch; the measured epoch took 26 minutes because the laptop
was on battery. After one epoch: test MSE 0.161, MAE 0.241.
