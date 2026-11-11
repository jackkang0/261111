"""
Flicker KPI — AI 偵測（路線 B：小型 CNN，吃原始 6 幀）
======================================================
輸入（預設 23 個通道）
  18 ch  時間殘差 r_t = frame_t - mean_t(frame)，除以整張圖殘差的 MAD
         （畫面內容被消掉，只留 pipeline 造成的擾動；數值是「相對噪聲底的倍率」）
   3 ch  6 幀平均影像的 R/G/B（內容本身）
   1 ch  平均影像的梯度能量（平坦區 / 邊緣紋理區）
   1 ch  局部噪聲底（16x16 區塊的殘差 MAD，取 log）
  後 5 個通道是必要的：同樣大小的殘差，在暗部與亮部、平坦區與紋理區的可見度不同，
  壓縮與 SR 的行為也跟局部內容有關。只給殘差的話網路無從判斷這些差異。

網路（約 4 千個參數，刻意很小，因為標記資料通常只有幾百個框）
      conv 3x3 stride2 (18->8)  ReLU
      conv 3x3 stride2 (8->16)  ReLU
      conv 3x3 stride2 (16->16) ReLU
      global max-pool + mean-pool -> 32 維 -> linear -> logit
  max-pool 是關鍵：120x120 的格子裡閃爍可能只占幾個 pixel，用平均會被稀釋。

實作全部以 NumPy 完成（含反向傳播與 Adam），訓練和推論都不需要額外套件。
"""

import numpy as np
import cv2

from flicker_kpi_core import imread_unicode

MODEL_VERSION = 1
DEFAULT_TILE = 120
RES_CLIP = 200.0        # 正規化殘差的上限（存成 float16 時上限是 65504）


# --------------------------------------------------------------------------
# 資料前處理
# --------------------------------------------------------------------------
def sequence_input(paths, content=True, eps=1e-6):
    """
    回傳 (X (C,H,W) float32, scale)
      C = 18（純殘差）或 23（殘差 + 內容 + 局部噪聲底）
    """
    frames = [imread_unicode(p)[:, :, ::-1].astype(np.float32) for p in paths]
    stack = np.stack(frames, axis=0)
    mean_img = stack.mean(axis=0)
    res = stack - mean_img[None]
    # 殘差的 MAD。靜態畫面 + dither 的素材有超過一半的殘差剛好是 0，
    # median 會等於 0；原本只加 eps=1e-6 當保護，除下去會放大到 1e7 量級，
    # 存成 float16（上限 65504）就變成 inf，之後 loss 立刻是 nan。
    # 以半個量化階當下限：小於 0.5 LSB 的殘差本來就沒有意義。
    mad = float(np.median(np.abs(res))) * 1.4826
    scale = max(mad, 0.5)
    # 再加一道保險：正規化後夾在 +-RES_CLIP（遠大於任何真實閃爍，但遠小於
    # float16 上限），單一壞點就無法污染整個 batch
    chans = [np.clip(res / scale, -RES_CLIP, RES_CLIP)
             .transpose(0, 3, 1, 2).reshape(-1, *res.shape[1:3])]
    if content:
        chans.append((mean_img.transpose(2, 0, 1) / 128.0 - 1.0))      # 內容 R/G/B
        g = (0.299 * mean_img[..., 0] + 0.587 * mean_img[..., 1]
             + 0.114 * mean_img[..., 2]).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        chans.append((np.log1p(np.abs(gx) + np.abs(gy)) / 5.0)[None])  # 梯度能量
        amp = np.abs(res).mean(axis=0).mean(axis=2)                    # 局部噪聲底
        H, W = amp.shape
        small = cv2.resize(amp, (max(1, W // 16), max(1, H // 16)),
                           interpolation=cv2.INTER_AREA)
        local = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        chans.append((np.log1p(local / scale))[None])
        # 內容通道各自標準化到與殘差相近的尺度，否則初期會被網路忽略
        extra = np.concatenate(chans[1:], axis=0)
        mu = extra.mean(axis=(1, 2), keepdims=True)
        sd = extra.std(axis=(1, 2), keepdims=True) + 1e-3
        chans[1:] = [np.clip((extra - mu) / sd, -4, 4)]
    return np.concatenate(chans, axis=0).astype(np.float32), scale


def sequence_residual(paths, eps=1e-6):
    """
    讀 6 幀 -> (N, H, W, 3) 時間殘差 / 噪聲尺度
    噪聲尺度取整張圖殘差的 MAD（中位數絕對偏差），對少數強閃爍不敏感。
    """
    frames = [imread_unicode(p)[:, :, ::-1].astype(np.float32) for p in paths]
    stack = np.stack(frames, axis=0)
    res = stack - stack.mean(axis=0, keepdims=True)
    scale = float(np.median(np.abs(res))) * 1.4826 + eps
    return (res / scale).astype(np.float32), scale


def tile_patches(res, tile=DEFAULT_TILE):
    """(N,H,W,3) -> (n_tile, N*3, tile, tile)；通道順序為 frame-major（舊介面）"""
    N, H, W, C = res.shape
    th, tw = H // tile, W // tile
    x = res[:, :th * tile, :tw * tile]
    x = x.reshape(N, th, tile, tw, tile, C)
    x = x.transpose(1, 3, 0, 5, 2, 4)
    return np.ascontiguousarray(x.reshape(th * tw, N * C, tile, tile)), (th, tw)


def split_tiles(X, tile=DEFAULT_TILE):
    """(C,H,W) -> (n_tile, C, tile, tile)"""
    C, H, W = X.shape
    th, tw = H // tile, W // tile
    x = X[:, :th * tile, :tw * tile].reshape(C, th, tile, tw, tile)
    x = x.transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(x.reshape(th * tw, C, tile, tile)), (th, tw)


def tile_labels(boxes, grid, tile=DEFAULT_TILE):
    th, tw = grid
    y = np.zeros(th * tw, np.int8)
    for (x0, y0, x1, y1) in boxes or ():
        for r in range(max(0, y0 // tile), min(th, -(-y1 // tile))):
            for c in range(max(0, x0 // tile), min(tw, -(-x1 // tile))):
                y[r * tw + c] = 1
    return y


# --------------------------------------------------------------------------
# 基本層（im2col 版卷積 + 反向傳播）
# --------------------------------------------------------------------------
def _im2col(x, k, s):
    n, c, h, w = x.shape
    oh, ow = (h - k) // s + 1, (w - k) // s + 1
    strd = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, c, oh, ow, k, k),
        strides=(strd[0], strd[1], strd[2] * s, strd[3] * s, strd[2], strd[3]))
    return view.reshape(n, c, oh, ow, k * k), (oh, ow)


def conv_forward(x, W, b, s):
    k = W.shape[-1]
    col, (oh, ow) = _im2col(x, k, s)
    n, c = x.shape[:2]
    col = col.transpose(0, 2, 3, 1, 4).reshape(n * oh * ow, c * k * k)
    out = col @ W.reshape(W.shape[0], -1).T + b
    return out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2), (col, x.shape, (oh, ow))


def conv_backward(dout, cache, W, s):
    col, xshape, (oh, ow) = cache
    n, c, h, w = xshape
    k = W.shape[-1]
    dout_t = dout.transpose(0, 2, 3, 1).reshape(n * oh * ow, -1)
    dW = (dout_t.T @ col).reshape(W.shape)
    db = dout_t.sum(axis=0)
    dcol = (dout_t @ W.reshape(W.shape[0], -1)).reshape(n, oh, ow, c, k * k)
    dx = np.zeros(xshape, np.float32)
    dcol = dcol.transpose(0, 3, 1, 2, 4)                      # (n,c,oh,ow,k*k)
    for i in range(k):                                        # k 很小，直接累加
        for j in range(k):
            dx[:, :, i:i + oh * s:s, j:j + ow * s:s] += dcol[..., i * k + j]
    return dx, dW, db


class TinyCNN:
    """conv-relu x3 -> (max, mean) global pool -> linear"""

    def __init__(self, in_ch, chans=(8, 16, 16), seed=0):
        rng = np.random.default_rng(seed)
        self.s = 2
        self.k = 3
        self.W, self.b = [], []
        prev = in_ch
        for c in chans:
            fan = prev * self.k * self.k
            self.W.append((rng.standard_normal((c, prev, self.k, self.k))
                           * np.sqrt(2.0 / fan)).astype(np.float32))
            self.b.append(np.zeros(c, np.float32))
            prev = c
        self.Wf = (rng.standard_normal(prev * 2) * 0.1).astype(np.float32)
        self.bf = np.float32(0.0)

    # ---- forward / backward ----
    def forward(self, x, cache=None):
        h = x
        caches = []
        for W, b in zip(self.W, self.b):
            h, cc = conv_forward(h, W, b, self.s)
            mask = h > 0
            h = h * mask
            caches.append((cc, mask))
        n, c = h.shape[:2]
        flat = h.reshape(n, c, -1)
        idx = flat.argmax(axis=2)
        mx = np.take_along_axis(flat, idx[:, :, None], axis=2)[:, :, 0]
        mean = flat.mean(axis=2)
        feat = np.concatenate([mx, mean], axis=1)
        z = feat @ self.Wf + self.bf
        if cache is not None:
            cache.update(caches=caches, feat=feat, idx=idx, hshape=h.shape)
        return z

    def backward(self, dz, cache):
        feat, idx, hshape = cache["feat"], cache["idx"], cache["hshape"]
        n, c, oh, ow = hshape
        dWf = feat.T @ dz
        dbf = dz.sum()
        dfeat = dz[:, None] * self.Wf[None, :]
        dmx, dmean = dfeat[:, :c], dfeat[:, c:]
        dflat = np.zeros((n, c, oh * ow), np.float32)
        np.put_along_axis(dflat, idx[:, :, None], dmx[:, :, None], axis=2)
        dflat += (dmean / (oh * ow))[:, :, None]
        dh = dflat.reshape(hshape)
        dW_all, db_all = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            cc, mask = cache["caches"][i]
            dh = dh * mask
            dh, dW, db = conv_backward(dh, cc, self.W[i], self.s)
            dW_all[i], db_all[i] = dW, db
        return dW_all, db_all, dWf, dbf

    # ---- 參數存取 ----
    def params(self):
        return self.W + self.b + [self.Wf, np.array([self.bf], np.float32)]

    def set_params(self, ps):
        n = len(self.W)
        self.W = [np.asarray(p, np.float32) for p in ps[:n]]
        self.b = [np.asarray(p, np.float32) for p in ps[n:2 * n]]
        self.Wf = np.asarray(ps[2 * n], np.float32)
        self.bf = float(np.asarray(ps[2 * n + 1]).ravel()[0])


# --------------------------------------------------------------------------
# 訓練
# --------------------------------------------------------------------------
def augment(x, rng):
    """翻轉、旋轉 90 度、幀順序反轉（閃爍對這些變換不變）"""
    if rng.random() < 0.5:
        x = x[:, :, :, ::-1]
    if rng.random() < 0.5:
        x = x[:, :, ::-1, :]
    r = int(rng.integers(0, 4))
    if r:
        x = np.rot90(x, r, axes=(2, 3))
    return np.ascontiguousarray(x)


def train(X, y, in_ch=None, epochs=40, batch=16, lr=3e-3, wd=1e-4, seed=0,
          chans=(8, 16, 16), val=None, log=None):
    """
    X: (n, C, t, t) float32；y: (n,) 0/1
    回傳 (model, history)
    """
    rng = np.random.default_rng(seed)
    net = TinyCNN(in_ch or X.shape[1], chans, seed)
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    y = np.asarray(y, np.float32)
    pos, neg = max(y.sum(), 1), max((1 - y).sum(), 1)
    w_pos, w_neg = 0.5 * len(y) / pos, 0.5 * len(y) / neg
    hist, step = [], 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb = augment(np.asarray(X[idx], np.float32), rng)
            yb = y[idx]
            cache = {}
            z = net.forward(xb, cache)
            wb = np.where(yb > 0, w_pos, w_neg).astype(np.float32)
            loss = float((wb * (np.logaddexp(0, z) - yb * z)).mean())
            dz = (wb * (1 / (1 + np.exp(-z)) - yb) / len(idx)).astype(np.float32)
            dW, db, dWf, dbf = net.backward(dz, cache)
            grads = dW + db + [dWf, np.array([dbf], np.float32)]
            ps = net.params()
            step += 1
            for j, (p, g) in enumerate(zip(ps, grads)):
                g = g + wd * p
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                mh = m[j] / (1 - 0.9 ** step)
                vh = v[j] / (1 - 0.999 ** step)
                ps[j] = p - lr * mh / (np.sqrt(vh) + 1e-8)
            net.set_params(ps)
            tot += loss * len(idx)
        rec = {"epoch": ep + 1, "loss": tot / len(X)}
        if val is not None:
            rec["val_auc"] = auc(predict(net, val[0]), val[1])
        hist.append(rec)
        if log:
            log(rec)
    return net, hist


def predict(net, X, batch=64):
    out = []
    for i in range(0, len(X), batch):
        out.append(net.forward(np.ascontiguousarray(X[i:i + batch], np.float32)))
    z = np.concatenate(out) if out else np.zeros(0)
    return 1.0 / (1.0 + np.exp(-z))


def _rank_avg(a):
    """平均排名（同分取平均）。np.argsort(np.argsort(.)) 只給序位排名，
    所有值相同時會依出現順序給 1..n，使 AUC 變成 0 而非 0.5。"""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    gid = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1      # 同分的歸為一組
    cnt = np.bincount(gid)
    ends = np.cumsum(cnt)
    avg = (ends - cnt + ends + 1) / 2.0                    # 該組佔的序位取平均
    r = np.empty(len(a), float)
    r[order] = avg[gid]
    return r


def auc(scores, y):
    y = np.asarray(y)
    s = np.asarray(scores, float)
    if not np.isfinite(s).all():        # 模型發散時分數會是 nan/inf
        return float("nan")
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = _rank_avg(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# --------------------------------------------------------------------------
# 存檔 / 載入
# --------------------------------------------------------------------------
# ==========================================================================
# 路線 B 的另一種結構：cnn3d（由 flicker_cnn_torch.py 訓練後匯出）
# ==========================================================================
# 推論時 BatchNorm 已經在匯出階段摺進前一層 conv 的權重，所以這裡只需要
# conv + ReLU + pooling，不必實作 BN：
#     y = (conv(x) - mean) / sqrt(var + eps) * gamma + beta
#       = conv(x; W * g, b * g + beta - mean * g)        g = gamma / sqrt(var + eps)


def _conv2d(x, W, b, stride=1, pad=0):
    """(N,C,H,W) 2D 卷積，支援 zero padding（_im2col 本身只做 valid）"""
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    return conv_forward(np.ascontiguousarray(x, np.float32), W, b, stride)[0]


def conv3d_k3t(x, W, b, stride_hw=1, pad_hw=0):
    """
    時間軸 kernel=3、pad=1、stride=1 的 Conv3d（輸出的 T 與輸入相同）。

    作法是沿時間軸拆成 3 個 2D 卷積再相加：
        out[t] = sum_{dt=0..2} conv2d(x[t + dt - 1], W[:, :, dt])
    時間軸超出範圍的部分補 0，與 torch 的 padding=(1,·,·) 一致。
    x: (B,C,T,H,W)   W: (O,C,3,kh,kw)   -> (B,O,T,oh,ow)
    """
    x = np.asarray(x, np.float32)
    B, C, T, H, Wd = x.shape
    O = W.shape[0]
    zero = np.zeros(O, np.float32)
    acc = None
    for dt in range(3):
        src = np.arange(T) + dt - 1                     # 輸出 t 取輸入 t+dt-1
        ok = (src >= 0) & (src < T)
        xs = np.zeros_like(x)
        xs[:, :, ok] = x[:, :, src[ok]]
        o = _conv2d(xs.transpose(0, 2, 1, 3, 4).reshape(B * T, C, H, Wd),
                    np.ascontiguousarray(W[:, :, dt]), zero, stride_hw, pad_hw)
        acc = o if acc is None else acc + o
    oh, ow = acc.shape[-2:]
    out = acc.reshape(B, T, O, oh, ow).transpose(0, 2, 1, 3, 4)
    return out + b.reshape(1, O, 1, 1, 1)


def _resize_nearest(x, hw):
    """(N,C,h,w) -> (N,C,H,W) 最近鄰，對應 torch 的 interpolate(mode='nearest')"""
    h, w = x.shape[-2:]
    H, W = hw
    if (h, w) == (H, W):
        return x
    ri = (np.arange(H) * h // H).clip(0, h - 1)
    ci = (np.arange(W) * w // W).clip(0, w - 1)
    return x[:, :, ri][:, :, :, ci]


class CNN3D:
    """flicker_cnn_torch.py 的 cnn3d，純 NumPy 推論版（BN 已摺入 conv）

    結構（與 torch 版一一對應）:
        stem     Conv3d(3  -> a, 3x3x3, stride (1,2,2), pad (1,1,1)) ReLU
                 Conv3d(a  -> b, 3x3x3, stride (1,2,2), pad (1,1,1)) ReLU
        temporal Conv3d(b  -> b, 3x1x1, pad (1,0,0))                 ReLU
                 時間軸取 max
        spatial  Conv2d(b+extra -> c, 3, stride 2, pad 1)            ReLU
                 Conv2d(c  -> c, 3, stride 2, pad 1)                 ReLU
        head     Linear(2c -> 1)  吃 [全域 max, 全域 mean]
    """

    def __init__(self, w):
        self.w = w          # {名稱: ndarray}

    def forward(self, x, extra=None):
        w = self.w
        h = x
        for i in (0, 1):                                  # stem
            h = conv3d_k3t(h, w[f"stem{i}_W"], w[f"stem{i}_b"], 2, 1)
            h = np.maximum(h, 0, out=h)
        h = conv3d_k3t(h, w["temporal_W"], w["temporal_b"], 1, 0)
        h = np.maximum(h, 0, out=h)
        h = h.max(axis=2)                                 # 收掉時間軸
        if extra is not None and extra.shape[1]:
            h = np.concatenate([h, _resize_nearest(extra, h.shape[-2:])], axis=1)
        for i in (0, 1):                                  # spatial
            h = _conv2d(h, w[f"sp{i}_W"], w[f"sp{i}_b"], 2, 1)
            h = np.maximum(h, 0, out=h)
        n, c = h.shape[:2]
        flat = h.reshape(n, c, -1)
        feat = np.concatenate([flat.max(axis=2), flat.mean(axis=2)], axis=1)
        return feat @ w["head_W"] + w["head_b"]


def split_3d(X, frames=6):
    """(n, C, t, t) -> ((n,3,T,t,t) 殘差, (n,extra,t,t) 內容通道或 None)"""
    X = np.asarray(X, np.float32)
    n, C, h, wd = X.shape
    res = X[:, :3 * frames].reshape(n, frames, 3, h, wd).transpose(0, 2, 1, 3, 4)
    extra = X[:, 3 * frames:] if C > 3 * frames else None
    return np.ascontiguousarray(res), (None if extra is None or not extra.shape[1]
                                       else np.ascontiguousarray(extra))


def predict3d(net, X, frames=6, batch=32):
    out = []
    for i in range(0, len(X), batch):
        res, extra = split_3d(X[i:i + batch], frames)
        out.append(net.forward(res, extra).ravel())
    z = np.concatenate(out) if out else np.zeros(0)
    return 1.0 / (1.0 + np.exp(-z))


# ==========================================================================
# 直接跑 PyTorch 的 .pt checkpoint（環境裝了 torch 才能用）
# ==========================================================================
# 有 torch 時完全不必先 export 成 .npz：架構由 flicker_cnn_torch.build_model
# 重建，權重直接載入，跟訓練時是同一份定義，不存在轉換誤差。
# 沒有 torch 的機器仍然走 .npz + 純 NumPy 推論。


def torch_available():
    try:
        import torch                                    # noqa: F401
        return True
    except Exception:
        return False


class TorchNet:
    """包住一個已載入的 torch 模型，提供與 NumPy 版一致的 predict 介面"""

    def __init__(self, model, arch, frames, device):
        self.model = model
        self.arch = arch
        self.frames = frames
        self.device = device

    def predict(self, X, batch=64):
        import torch
        import flicker_cnn_torch as ft
        out = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(
                    np.ascontiguousarray(X[i:i + batch], np.float32)).to(self.device)
                inp, extra = ft.to_model_input(xb, self.arch, self.frames)
                z = self.model(inp) if extra is None else self.model(inp, extra)
                out.append(torch.sigmoid(z).cpu().numpy().ravel())
        return np.concatenate(out) if out else np.zeros(0)


def load_torch(path, device=None):
    """載入 flicker_cnn_torch.py 存的 .pt，回傳 (TorchNet, meta)"""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "這個檔案是 PyTorch 的 .pt 模型，但目前環境沒有安裝 torch。\n"
            "請改用在訓練機上轉出的 .npz（純 NumPy 推論、不需要 torch）：\n"
            "    python flicker_cnn_torch.py export --model 這個.pt --out 模型.npz"
        ) from None
    import flicker_cnn_torch as ft
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(ck["meta"])
    arch = meta.get("arch", "tiny2d")
    frames = int(meta.get("frames", 6))
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ft.build_model(arch, int(meta["in_ch"]), frames)
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    meta["kind"] = "cnn"
    meta["engine"] = f"pytorch ({dev.type})"
    return TorchNet(model, arch, frames, dev), meta


def save(path, net, meta):
    arrs = {f"p{i}": p for i, p in enumerate(net.params())}
    import json
    meta = dict(meta)
    meta.setdefault("arch", "tiny2d")        # 明確標註，載入時不必靠「沒有 arch」推斷
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), n=len(net.W), **arrs)
    return path


CNN3D_KEYS = ("stem0_W", "stem0_b", "stem1_W", "stem1_b",
              "temporal_W", "temporal_b", "sp0_W", "sp0_b", "sp1_W", "sp1_b",
              "head_W", "head_b")


def load(path):
    """回傳 (net, meta)。meta["arch"] == "cnn3d" 時是 CNN3D，否則是 TinyCNN"""
    import json
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    if meta.get("arch") == "cnn3d":
        missing = [k for k in CNN3D_KEYS if k not in d.files]
        if missing:
            raise ValueError(
                f"這個檔案標示 arch=cnn3d，但缺少 cnn3d 的權重（{', '.join(missing[:3])}…）。"
                "檔案可能是 tiny2d 卻被改過 meta，或匯出時中斷了，請重新匯出。")
        w = {k: np.ascontiguousarray(d[k], np.float32) for k in CNN3D_KEYS}
        net = CNN3D(w)
        _selftest(net, d, meta)
        return net, meta
    if "n" not in d.files:
        raise ValueError(
            "無法辨識這個模型檔：既不是 cnn3d（meta 沒有 arch=cnn3d），"
            "也沒有 tiny2d 的權重。請確認檔案是由 flicker_cnn_torch.py export "
            "或 GUI 的「儲存模型」產生的。")
    n = int(d["n"])
    ps = [d[f"p{i}"] for i in range(2 * n + 2)]
    net = TinyCNN(ps[0].shape[1], tuple(p.shape[0] for p in ps[:n]))
    meta.setdefault("arch", "tiny2d")
    net.set_params(ps)
    return net, meta


def _selftest(net, d, meta, tol=2e-3):
    """匯出時若附了 PyTorch 的參考輸出，載入時比對一次

    這是刻意加的保險：匯出端（torch）與推論端（NumPy）是兩份獨立實作，
    權重排列或 BatchNorm 摺入只要有一點對不上，分數就會整個錯掉而且沒有徵兆。
    """
    if "selftest_x" not in d or "selftest_y" not in d:
        meta["selftest"] = "模型檔沒有附參考輸出，無法驗證"
        return
    x = np.asarray(d["selftest_x"], np.float32)
    want = np.asarray(d["selftest_y"], np.float32).ravel()
    got = predict3d(net, x, int(meta.get("frames", 6)))
    err = float(np.abs(got - want).max())
    meta["selftest"] = f"與 PyTorch 的最大差 {err:.2e}"
    if not (err < tol):
        raise ValueError(
            f"cnn3d 自我檢查失敗：NumPy 推論與 PyTorch 的輸出最大相差 {err:.3e}"
            f"（容許 {tol:g}）。\n參考值 {np.round(want[:4], 4)}"
            f"  本地值 {np.round(got[:4], 4)}\n"
            "這代表匯出的權重與推論實作對不上，請重新匯出；若仍失敗請回報。")


def score_tiles(net, paths, tile=DEFAULT_TILE, content=True, frames=6):
    """回傳 (分數圖 (th,tw), grid)；自動分辨 TinyCNN 與 CNN3D"""
    arr, _ = sequence_input(paths, content)
    X, grid = split_tiles(arr, tile)
    if isinstance(net, TorchNet):
        sc = net.predict(X)
    elif isinstance(net, CNN3D):
        sc = predict3d(net, X, frames)
    else:
        sc = predict(net, X)
    return sc.reshape(grid), grid
