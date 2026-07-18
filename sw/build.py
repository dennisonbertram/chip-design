#!/usr/bin/env python3
"""build.py - train, quantize, compile, and golden-simulate the nano-LM for nano_accel.

Pipeline:
  1. fetch corpus (tiny-shakespeare; fallback: local word list)
  2. train char-level MLP (context 8, embed 24, hidden 512 x2, vocab 128) in numpy
  3. quantize: INT4 weights (per-output-channel symmetric), INT8 activations
  4. run the golden INTEGER model that mirrors rtl/nano_accel.sv bit-for-bit
  5. emit out/*.hex: wsram, esram, bsram, msram, tsram (prompt), isram (program),
     expected.hex (golden tokens) for the testbench

Numerics contract with the RTL (must stay in sync):
  acc  = sum(w_int4 * x_int8) + b_int32                    (INT32, exact)
  pb   = acc * M[c]                                        (INT48)
  y    = (pb + (1 << (sh-1))) >> sh   [arithmetic shift; if sh>0]
  y    = relu?(y); y = clip(y, -128, 127)                  (INT8)
  head layer: argmax over pb (no requant)
"""
import os, sys, subprocess, time
import numpy as np

V, E, C, H = 128, 24, 8, 512
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
PROMPT_ARG = sys.argv[2] if len(sys.argv) > 2 else None
G = int(sys.argv[3]) if len(sys.argv) > 3 else 32   # generated tokens (tb must match)
BATCH = 256
SEED = 1234
rng = np.random.default_rng(SEED)
os.makedirs("out", exist_ok=True)

# ---------------------------------------------------------------- corpus
CORPUS = "out/corpus.txt"
if not os.path.exists(CORPUS):
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
           "master/data/tinyshakespeare/input.txt")
    try:
        subprocess.run(["curl", "-sL", "--max-time", "20", url, "-o", CORPUS],
                       check=True)
        assert os.path.getsize(CORPUS) > 100_000
        print("[corpus] downloaded tiny-shakespeare")
    except Exception:
        print("[corpus] download failed; falling back to local word list")
        with open("/usr/share/dict/words") as f:
            words = f.read()
        with open(CORPUS, "w") as f:
            f.write((words + "\n") * 8)

text = open(CORPUS, encoding="ascii", errors="ignore").read()
text = "".join(ch if ord(ch) < 128 else " " for ch in text)
ids = np.frombuffer(text.encode("ascii"), dtype=np.uint8).astype(np.int64)
print(f"[corpus] {len(ids)} bytes")

def get_batch(bs):
    s = rng.integers(0, len(ids) - C - 1, bs)
    X = ids[s[:, None] + np.arange(C)]
    Y = ids[s + C]
    return X, Y

# ---------------------------------------------------------------- model
def init(n_in, n_out):
    return (rng.standard_normal((n_in, n_out)) * (1.0 / n_in) ** 0.5).astype(np.float64)

EMB = init(V, E); W1 = init(C * E, H); b1 = np.zeros(H)
W2 = init(H, H);  b2 = np.zeros(H)
W3 = init(H, V);  b3 = np.zeros(V)
params = [EMB, W1, b1, W2, b2, W3, b3]

def forward(X):
    x = EMB[X].reshape(len(X), C * E)
    h1 = np.maximum(x @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    z = h2 @ W3 + b3
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    return x, h1, h2, p

# ---------------------------------------------------------------- train
m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
lr, b1a, b2a, eps = 3e-3, 0.9, 0.999, 1e-8
t0 = time.time()
for step in range(1, STEPS + 1):
    X, Y = get_batch(BATCH)
    x, h1, h2, p = forward(X)
    loss = -np.log(p[np.arange(len(X)), Y] + 1e-12).mean()
    dz = p; dz[np.arange(len(X)), Y] -= 1; dz /= len(X)
    dW3 = h2.T @ dz; db3 = dz.sum(0)
    dh2 = dz @ W3.T; dh2[h2 <= 0] = 0
    dW2 = h1.T @ dh2; db2 = dh2.sum(0)
    dh1 = dh2 @ W2.T; dh1[h1 <= 0] = 0
    dW1 = x.T @ dh1; db1 = dh1.sum(0)
    dx = dh1 @ W1.T
    dEMB = np.zeros_like(EMB)
    np.add.at(dEMB, X, dx.reshape(len(X), C, E))
    grads = [dEMB, dW1, db1, dW2, db2, dW3, db3]
    for i in range(len(params)):
        m[i] = b1a * m[i] + (1 - b1a) * grads[i]
        v[i] = b2a * v[i] + (1 - b2a) * grads[i] ** 2
        mh = m[i] / (1 - b1a ** step); vh = v[i] / (1 - b2a ** step)
        params[i] -= lr * mh / (np.sqrt(vh) + eps)
    if step % 500 == 0 or step == 1:
        print(f"[train] step {step:5d}  loss {loss:.3f}  ({time.time()-t0:.0f}s)")
EMB, W1, b1, W2, b2, W3, b3 = params
print(f"[train] done in {time.time()-t0:.0f}s, final loss {loss:.3f}")

# ------------------------------------------------------------- quantize
def quant_w(W):                       # per-output-channel symmetric INT4
    s = np.abs(W).max(axis=0) / 7.0
    s[s == 0] = 1.0
    q = np.clip(np.round(W / s), -8, 7).astype(np.int64)
    return q, s

def pick_sh(ratio):                   # drain shift so max M fits in 15 bits
    return max(1, int(np.floor(np.log2(32767.0 / ratio.max()))))

W1q, sW1 = quant_w(W1); W2q, sW2 = quant_w(W2); W3q, sW3 = quant_w(W3)

s_x1 = np.abs(EMB).max() / 127.0                     # L1 input scale (static)
emb_q = np.clip(np.round(EMB / s_x1), -127, 127).astype(np.int64)

def golden_mv(xq, Wq, bq, M, sh, relu):              # mirrors RTL drain exactly
    acc = Wq.astype(np.int64).T @ xq.astype(np.int64) + bq
    pb = acc * M
    y = (pb + (1 << (sh - 1))) >> sh if sh > 0 else pb
    if relu: y = np.maximum(y, 0)
    return np.clip(y, -128, 127)

# calibration windows
Xs, _ = get_batch(4096)
x_cal = emb_q[Xs].reshape(len(Xs), C * E)

# --- layer 1: choose s_x2 with margin until no clipping on calibration data
b1q = np.round(b1 / (sW1 * s_x1)).astype(np.int64)
margin = 1.25
while True:
    h1_f = np.maximum(x_cal @ (W1q * sW1 * s_x1) + b1, 0)
    s_x2 = h1_f.max() * margin / 127.0
    r1 = sW1 * s_x1 / s_x2
    sh1 = pick_sh(r1); M1 = np.clip(np.round(r1 * 2**sh1), 0, 32767).astype(np.int64)
    h1q = golden_mv(x_cal[0], W1q, b1q, M1, sh1, True)
    clip = 0
    for i in range(0, 4096, 16):
        acc = (W1q.T @ x_cal[i] + b1q) * M1
        clip += int(((acc + (1 << (sh1-1))) >> sh1 > 127).sum())
    if clip == 0: break
    margin *= 1.15
print(f"[quant] L1: s_x2={s_x2:.5f} sh1={sh1} clip_events={clip}")

# --- layer 2
H1q = np.stack([golden_mv(x_cal[i], W1q, b1q, M1, sh1, True)
                for i in range(0, 4096, 4)])
b2q = np.round(b2 / (sW2 * s_x2)).astype(np.int64)
margin = 1.25
while True:
    h2_f = np.maximum((H1q * s_x2) @ (W2q * sW2) + b2, 0)
    s_x3 = h2_f.max() * margin / 127.0
    r2 = sW2 * s_x2 / s_x3
    sh2 = pick_sh(r2); M2 = np.clip(np.round(r2 * 2**sh2), 0, 32767).astype(np.int64)
    clip = 0
    for i in range(len(H1q)):
        acc = (W2q.T @ H1q[i] + b2q) * M2
        clip += int(((acc + (1 << (sh2-1))) >> sh2 > 127).sum())
    if clip == 0: break
    margin *= 1.15
print(f"[quant] L2: s_x3={s_x3:.5f} sh2={sh2} clip_events={clip}")

# --- head
H2q = np.stack([golden_mv(H1q[i], W2q, b2q, M2, sh2, True)
                for i in range(0, len(H1q), 4)])
b3q = np.round(b3 / (sW3 * s_x3)).astype(np.int64)
r3 = sW3 * s_x3
sh3 = pick_sh(r3); M3 = np.clip(np.round(r3 * 2**sh3), 0, 32767).astype(np.int64)
print(f"[quant] HEAD: sh3={sh3}")
for nm, bq in (("b1", b1q), ("b2", b2q), ("b3", b3q)):
    assert np.abs(bq).max() < 2**30, f"{nm} overflow"
pb_all = (H2q @ W3q.astype(np.int64) + b3q) * M3
pbmax = int(np.abs(pb_all).max())
assert pbmax < 2**47, "logit pb overflow"
print(f"[quant] max |logit pb| = {pbmax:.3e} (< 2^47 ok)")

# ------------------------------------------------- golden autoregressive gen
PROMPT = PROMPT_ARG if PROMPT_ARG is not None else \
    ("First Ci" if text.startswith("First Ci") else text[:8])
assert len(PROMPT) == C and all(ord(c) < 128 for c in PROMPT), \
    f"prompt must be exactly {C} ASCII chars, got {PROMPT!r}"

def gen_tokens_int(prompt, n):
    toks = [ord(c) for c in prompt]
    for _ in range(n):
        xq = emb_q[toks[-C:]].reshape(C * E)
        h1 = golden_mv(xq, W1q, b1q, M1, sh1, True)
        h2 = golden_mv(h1, W2q, b2q, M2, sh2, True)
        pb = (W3q.astype(np.int64).T @ h2 + b3q) * M3   # 48b compare, like RTL
        toks.append(int(np.argmax(pb)))
    return toks

def gen_tokens_float(prompt, n):
    toks = [ord(c) for c in prompt]
    for _ in range(n):
        x = EMB[toks[-C:]].reshape(C * E)
        h1 = np.maximum(x @ W1 + b1, 0); h2 = np.maximum(h1 @ W2 + b2, 0)
        toks.append(int(np.argmax(h2 @ W3 + b3)))
    return toks

gi = gen_tokens_int(PROMPT, G); gf = gen_tokens_float(PROMPT, G)
si = "".join(chr(t) for t in gi[C:]); sf = "".join(chr(t) for t in gf[C:])
agree = np.mean([a == b for a, b in zip(gi[C:], gf[C:])])
print(f"[gen] prompt    = {PROMPT!r}")
print(f"[gen] float     = {sf!r}")
print(f"[gen] int4 chip = {si!r}")
print(f"[gen] float/int4 token agreement: {agree*100:.0f}%")

# ------------------------------------------------------------ emit hex
def w4(v): return int(v) & 0xF
def b8(v): return int(v) & 0xFF

def emit_wsram():
    words = []
    for Wq in (W1q, W2q, W3q):                    # rows = output channels
        for c in range(Wq.shape[1]):
            row = Wq[:, c]
            for k in range(0, len(row), 64):
                v = 0
                for i, w in enumerate(row[k:k+64]):
                    v |= w4(w) << (4 * i)
                words.append("%064x" % v)
    return words

def emit_esram():
    flat = [b8(v) for t in range(V) for v in emb_q[t]]
    return ["%08x" % (flat[i] | flat[i+1] << 8 | flat[i+2] << 16 | flat[i+3] << 24)
            for i in range(0, len(flat), 4)]

def emit_bsram():
    return ["%08x" % (int(v) & 0xFFFFFFFF)
            for v in np.concatenate([b1q, b2q, b3q])]

def emit_msram():
    return ["%04x" % int(v) for v in np.concatenate([M1, M2, M3])]

def instr(op, flags=0, sh=0, n=0, m=0, a0=0, a1=0, a2=0, a3=0, a4=0):
    v = (op | flags << 4 | sh << 8 | n << 16 | m << 32 | a0 << 48 |
         a1 << 64 | a2 << 80 | a3 << 96 | a4 << 112)
    return "%032x" % v

def emit_isram():
    prog = [instr(1, a0=C)]                       # SETLEN 8
    for _ in range(G):
        prog += [
            instr(2, n=C, m=E, a0=0, a1=0),                   # GATHER -> X@0
            instr(3, flags=1, sh=sh1, n=C*E, m=H,             # L1 -> H1@4
                  a0=0, a1=0, a2=4, a3=0, a4=0),
            instr(3, flags=1, sh=sh2, n=H, m=H,               # L2 -> H2@12
                  a0=1536, a1=4, a2=12, a3=512, a4=512),
            instr(3, flags=2, sh=sh3, n=H, m=V,               # HEAD -> LBUF
                  a0=5632, a1=12, a3=1024, a4=1024),
            instr(4, m=V),                                    # ARGMAX append
        ]
    prog += [instr(7)]                                        # HALT
    assert len(prog) <= 256
    return prog

def dump(name, lines):
    with open(f"out/{name}", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[emit] out/{name}: {len(lines)} words")

dump("wsram.hex", emit_wsram())
dump("esram.hex", emit_esram())
dump("bsram.hex", emit_bsram())
dump("msram.hex", emit_msram())
dump("isram.hex", emit_isram())
dump("tsram.hex", ["%02x" % ord(c) for c in PROMPT])
dump("expected.hex", ["%02x" % t for t in gi[C:]])

nw = W1q.size + W2q.size + W3q.size
print(f"[done] params={nw} ({nw*4/8/1024:.0f} KiB INT4), prompt={PROMPT!r}")
