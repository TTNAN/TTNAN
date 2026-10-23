"""ASCII koi pond, after stackedlol/ascii-sharks' renderer (spine + body wave,
supersampled cells, shaded blocks with density glyphs, diffusing wake).
Renders a seamless loop of frames to ./frames/."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hframes")
os.makedirs(OUT, exist_ok=True)

CW, CH = 8, 12
COLS, ROWS = 128, 70
W, H = COLS * CW, ROWS * CH
T = float(os.environ.get("LOOP", 28))
FPS = int(os.environ.get("FPS", 12))
NF = int(T * FPS)
N = 24  # spine nodes
FONT = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DejaVuSansMono-Bold.ttf"), int(CH * 0.78))

SKIN_RAMP = ":;t%SX8#@"
FIN_GLYPHS = "~-:;'"
EDGE_HEAVY = ":;"
EDGE_LIGHT = ".',`"

def h3(x, y, z):
    x = np.asarray(x, dtype=np.int64) & 0xFFFFFFFF
    y = np.asarray(y, dtype=np.int64) & 0xFFFFFFFF
    z = np.asarray(z, dtype=np.int64) & 0xFFFFFFFF
    h = ((x * 374761393) ^ (y * 668265263) ^ (z * 1442695041)) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFFFF) / 4294967296.0

def vnoise(x, y, seed):
    xi, yi = np.floor(x), np.floor(y)
    fx, fy = x - xi, y - yi
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    a = h3(xi, yi, seed); b = h3(xi + 1, yi, seed)
    c = h3(xi, yi + 1, seed); d = h3(xi + 1, yi + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

# ── koi silhouette (fractions of length, s: 0 snout → 1 tail tip) ──
def body_hw(s):
    s = np.asarray(s, dtype=float)
    head = 0.1 * np.power(np.sin(np.clip(s / 0.13, 0, 1) * np.pi / 2), 0.55)
    sh = 0.1 + 0.022 * smooth(0.13, 0.3, s)
    taper = 0.026 + 0.096 * np.power(np.clip(1 - (s - 0.3) / 0.46, 0, 1), 1.35)
    w = np.where(s < 0.13, head, np.where(s < 0.3, sh, np.where(s < 0.76, taper, 0.026 * (1 - smooth(0.76, 0.82, s)))))
    return w

def tail_hw(s, flare):
    k = np.clip((s - 0.74) / 0.26, 0, 1)
    return np.where(s > 0.74, 0.02 + (0.13 * flare) * np.power(np.sin(k * np.pi / 2), 0.9), 0)

WHITE = np.array([236, 234, 226.0]); RED = np.array([222, 58, 32.0])
BLACK = np.array([28, 28, 34.0]); GOLD = np.array([238, 176, 52.0])
ORANGE = np.array([240, 120, 30.0]); PLAT = np.array([214, 220, 228.0])

def pattern(kind, seed, s, vn):
    """Skin colour per sample (s along, vn across -1..1)."""
    n = len(s)
    out = np.tile(WHITE, (n, 1))
    blob = vnoise(s * 9, vn * 2.2 + 0.5, seed)
    blob2 = vnoise(s * 18 + 3, vn * 4 + 7, seed + 5)
    back = np.abs(vn) < 0.82
    if kind == "kohaku":
        red = (blob > 0.5 - 0.25 * (s < 0.13)) & back & (s > 0.02) & (s < 0.74)
        out[red] = RED
    elif kind == "sanke":
        red = (blob > 0.55) & back & (s > 0.03) & (s < 0.74)
        out[red] = RED
        blk = (blob2 > 0.78) & (s > 0.15) & (s < 0.7)
        out[blk] = BLACK
    elif kind == "showa":
        out[:] = BLACK
        red = (blob > 0.48) & (s < 0.74)
        out[red] = RED
        wht = (blob2 > 0.62) & ~red & (np.abs(vn) > 0.25)
        out[wht] = WHITE
    elif kind == "tancho":
        d = np.hypot((s - 0.085) / 0.05, vn / 0.55)
        out[d < 1] = RED
    elif kind == "ogon":
        out[:] = GOLD
    elif kind == "kujaku":
        out[:] = PLAT
        org = (blob > 0.45) & (s < 0.74)
        out[org] = ORANGE
        # net-like scale reticulation
        net = (np.abs(np.sin(s * 140) * np.sin(vn * 9)) > 0.93) & (s > 0.15) & (s < 0.72)
        out[net] = out[net] * 0.55
    return out

def fin_colour(kind):
    return {"ogon": GOLD * 0.95, "showa": np.array([200, 200, 205.0]), "kujaku": np.array([230, 170, 110.0])}.get(kind, np.array([225, 225, 230.0]))

# ── closed swim paths (koi leave frame and come back) ──
def make_path(spec):
    cx, cy, ax, ay, harm, dirn, phase = spec
    th = np.linspace(0, 2 * np.pi, 4001)
    x = cx + ax * np.cos(th)
    y = cy + ay * np.sin(th)
    for (k, bx, by, ph) in harm:
        x = x + bx * np.cos(k * th + ph)
        y = y + by * np.sin(k * th + ph)
    y = (y - 270) * (H / 540) + H / 2
    if dirn < 0:
        x, y = x[::-1], y[::-1]
    pts = np.stack([x, y], 1)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    arc = np.concatenate([[0], np.cumsum(seg)])
    return pts, arc

def path_at(path, a):
    pts, arc = path
    P = arc[-1]
    a = np.mod(a, P)
    x = np.interp(a, arc, pts[:, 0]); y = np.interp(a, arc, pts[:, 1])
    return x, y

KOI = [
    ("kohaku", 400, 1.00, (500, 280, 470, 250, [(2, 60, 40, 0.5)], 1, 0.0), 1, 26, 11),
    ("showa", 350, 0.88, (530, 260, 520, 240, [(3, 30, 40, 1.2)], -1, 1.9), 1, 28, 41),
    ("ogon", 320, 0.80, (510, 270, 400, 270, [(2, 80, 30, 2.0)], 1, 3.3), 1, 30, 37),
    ("tancho", 300, 0.74, (520, 265, 560, 220, [(2, 30, 70, 0.2)], -1, 4.6), 1, 30, 53),
    ("kujaku", 290, 0.68, (500, 275, 450, 255, [(3, 50, 35, 2.7)], 1, 5.5), 1, 31, 67),
]
PATHS = [make_path(k[3]) for k in KOI]
# start offsets so koi are spread round their loops
OFFS = [0.0, 0.62, 0.33, 0.12, 0.81]

def spine_for(i, t):
    kind, L, depth, spec, laps, beats, seed = KOI[i]
    path = PATHS[i]
    P = path[1][-1]
    a0 = (OFFS[i] + laps * t / T) * P
    ss = np.linspace(0, 1, N)
    x, y = path_at(path, a0 - ss * L)  # node 0 = snout, trailing behind along path
    nodes = np.stack([x, y], 1)
    # body wave growing toward tail
    tan = np.zeros_like(nodes)
    tan[1:-1] = nodes[:-2] - nodes[2:]
    tan[0] = nodes[0] - nodes[1]; tan[-1] = nodes[-2] - nodes[-1]
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    phase = 2 * np.pi * beats * t / T
    amp = L * 0.055 * (0.08 + 0.92 * ss ** 1.8)
    off = amp * np.sin(phase - ss * 5.0)
    sp = nodes.copy()
    sp[:, 0] -= tan[:, 1] * off
    sp[:, 1] += tan[:, 0] * off
    # turning rate for fin banking
    xa, ya = path_at(path, a0 + 8); xb, yb = path_at(path, a0 - 8)
    xc, yc = path_at(path, a0 - 24)
    h1 = math.atan2(ya - yb, xa - xb); h2 = math.atan2(yb - yc, xb - xc)
    turn = (h1 - h2 + math.pi) % (2 * math.pi) - math.pi
    return sp, phase, turn

def in_ellipse(u, a, cu, ca, ru, ra, ang):
    du, da = u - cu, a - ca
    c, s_ = math.cos(ang), math.sin(ang)
    x = du * c + da * s_; y = -du * s_ + da * c
    return (x / ru) ** 2 + (y / ra) ** 2 <= 1

FLAG_BODY, FLAG_EDGE, FLAG_FIN, FLAG_EYE, FLAG_RIDGE = 1, 2, 3, 4, 5

def sample(sp, L, P, kind, seed, phase, turn):
    """Classify points P (M,2) against a koi. Returns hit, flag, shade, s, v, rgb."""
    A = sp[:-1]; B = sp[1:]
    S = B - A
    len2 = (S ** 2).sum(1) + 1e-9
    seglen = np.sqrt(len2)
    arc = np.concatenate([[0], np.cumsum(seglen)])
    tot = arc[-1]
    d = P[:, None, :] - A[None]
    uraw = (d * S[None]).sum(2) / len2[None]
    u = np.clip(uraw, 0, 1)
    q = A[None] + S[None] * u[..., None]
    d2 = ((P[:, None, :] - q) ** 2).sum(2)
    k = d2.argmin(1)
    m = np.arange(len(P))
    dist = np.sqrt(d2[m, k])
    uk = u[m, k]
    past_tail = (k == len(S) - 1) & (uraw[m, k] > 1)
    s = (arc[k] + uk * seglen[k]) / tot
    cross = S[k, 0] * (P[:, 1] - A[k, 1]) - S[k, 1] * (P[:, 0] - A[k, 0])
    sign = np.where(cross >= 0, 1.0, -1.0)
    nx = -S[k, 1] / seglen[k] * sign; ny = S[k, 0] / seglen[k] * sign
    v = dist * sign
    us = s * L
    w = body_hw(s) * L
    flare = 1 + 0.12 * math.sin(phase * 0.5)
    tw = tail_hw(s, flare) * L

    hit = np.zeros(len(P), bool)
    flag = np.zeros(len(P), np.int8)
    shade = np.zeros(len(P))
    across = dist / np.maximum(w, 1e-6)
    onb = (dist <= w) & (s < 0.8)
    facing = (nx * -0.6 + ny * -0.8) * across
    sh_b = 0.62 - across ** 2 * 0.42 + facing * 0.24
    hit |= onb; flag[onb] = FLAG_BODY; shade[onb] = sh_b[onb]
    ridge = onb & (s > 0.3) & (s < 0.62) & (across < 0.13)
    flag[ridge] = FLAG_RIDGE
    eye = onb & (np.abs(s - 0.095) < 0.018) & (across > 0.58) & (across < 0.9)
    flag[eye] = FLAG_EYE; shade[eye] = 0

    # tail fan, forked, translucent
    tail = ~hit & (s > 0.74) & (dist <= tw) & ~past_tail
    fork = (s > 0.93) & (dist < tw * 0.45 * (s - 0.93) / 0.07)
    tail &= ~fork
    hit |= tail; flag[tail] = FLAG_FIN
    shade[tail] = (0.45 + 0.35 * (1 - np.clip(dist / np.maximum(tw, 1e-6), 0, 1)))[tail]

    # pectoral paddles, banking into turns and flapping
    bank = max(-0.3, min(0.3, turn * 2.5))
    flap = math.sin(phase * 0.5)
    rest = ~hit
    a = dist
    for side in (1.0, -1.0):
        sel = rest & (sign == side)
        reach = 1 + 0.18 * flap * side * 0.5 - bank * side
        pe = sel & in_ellipse(us, a, L * 0.26, L * (0.095 + 0.05 * reach), L * 0.07, L * 0.05 * reach, -0.55)
        hit |= pe; flag[pe] = FLAG_FIN
        shade[pe] = (0.5 + 0.3 * (1 - np.clip((a - L * 0.09) / (L * 0.1), 0, 1)))[pe]
        pv = sel & ~pe & in_ellipse(us, a, L * 0.55, L * 0.05, L * 0.04, L * 0.028, -0.5)
        hit |= pv; flag[pv] = FLAG_FIN; shade[pv] = 0.5
    vn = np.clip(v / np.maximum(w, 1e-6), -1, 1)
    rgb = pattern(kind, seed, s, vn)
    fin = flag == FLAG_FIN
    if fin.any():
        rgb[fin] = fin_colour(kind)
    return hit, flag, shade, s, v, rgb

SUB = np.array([[0.25, 0.3], [0.75, 0.3], [0.25, 0.72], [0.75, 0.72]])

# lily pads: (col, row, radius in px, notch angle, seed)
PADS = []

def render(fi, energy, draw_img=True):
    t = fi / FPS
    nc = COLS * ROWS
    cflag = np.zeros(nc, np.int8); cshade = np.zeros(nc); ckey = np.zeros(nc, np.int64)
    crgb = np.zeros((nc, 3)); cdim = np.ones(nc)
    order = sorted(range(len(KOI)), key=lambda i: KOI[i][2])
    tails = []
    for i in order:
        kind, L, depth, spec, laps, beats, seed = KOI[i]
        sp, phase, turn = spine_for(i, t)
        tails.append(sp[-8:])
        if not draw_img:
            continue
        reach = L * 0.25
        c0 = max(0, int((sp[:, 0].min() - reach) // CW)); c1 = min(COLS - 1, int((sp[:, 0].max() + reach) // CW))
        r0 = max(0, int((sp[:, 1].min() - reach) // CH)); r1 = min(ROWS - 1, int((sp[:, 1].max() + reach) // CH))
        if c0 > c1 or r0 > r1:
            continue
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        cc = cc.ravel(); rr = rr.ravel()
        base = np.stack([cc * CW, rr * CH], 1).astype(float)
        cen = base + [CW / 2, CH / 2]
        hit, flag, shade, s, v, rgb = sample(sp, L, cen, kind, seed, phase, turn)
        hits = hit.astype(int)
        for o in SUB:
            hs = sample(sp, L, base + o * [CW, CH], kind, seed, phase, turn)[0]
            hits += hs
        cov = hits / 5
        idx = rr * COLS + cc
        keys = (h3(np.round(s * 60), np.round(v / (CW * 0.9)), seed) * 4294967295).astype(np.int64)
        full = hit & (cov >= 0.6)
        edge = (hits > 0) & ~full
        tick = fi // 2
        flick = (h3(cc, rr, tick) - 0.5) * 0.06
        sel = full
        cflag[idx[sel]] = flag[sel]
        cshade[idx[sel]] = np.clip(shade[sel] + flick[sel], 0, 1)
        crgb[idx[sel]] = rgb[sel]
        ckey[idx[sel]] = keys[sel]
        cdim[idx[sel]] = 0.62 + 0.38 * depth
        e = edge & (cflag[idx] == 0)
        cflag[idx[e]] = FLAG_EDGE
        cshade[idx[e]] = cov[e]
        crgb[idx[e]] = rgb[e] if True else 0
        ckey[idx[e]] = keys[e]
        cdim[idx[e]] = 0.62 + 0.38 * depth

    # wake: tail sweeps water, diffuses and fades
    E = energy.reshape(ROWS, COLS)
    for tl in tails:
        for x, y in tl:
            c, r = int(x // CW), int(y // CH)
            if 0 <= c < COLS and 0 <= r < ROWS:
                E[r, c] = min(1.6, E[r, c] + 0.07)
    dt = 1 / FPS
    pad = np.pad(E, 1, mode="edge")
    E2 = (E * 4 + pad[1:-1, :-2] + pad[1:-1, 2:] + pad[:-2, 1:-1] + pad[2:, 1:-1]) * 0.125 * math.exp(-dt * 1.3)
    energy[:] = E2.ravel()
    if not draw_img:
        return None


    return draw_frame(fi, t, energy, cflag, cshade, ckey, crgb, cdim)

# ── static content: title, icons, motto ──
HERE = os.path.dirname(os.path.abspath(__file__))
SERIF = os.path.join(HERE, "lmroman10-italic.otf")
def text_mask(txt, width, y, size):
    f = ImageFont.truetype(SERIF, size)
    while f.getlength(txt) > width:
        size -= 1; f = ImageFont.truetype(SERIF, size)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).text((W / 2, y), txt, font=f, fill=255, anchor="mm")
    return np.asarray(m, dtype=float) / 255
TITLE = text_mask("Full-Stack Indie Developer | Open Source Builder", 930, 70, 46)
MOTTO = text_mask("Fortune favors the bold.", 520, 470, 46)
TEXT = np.maximum(TITLE, MOTTO)
STOPS = [(0, (154, 123, 30)), (0.3, (212, 175, 55)), (0.5, (255, 243, 176)), (0.7, (212, 175, 55)), (1, (154, 123, 30))]
def gold_row(shift):
    x = np.arange(W) / W - shift
    out = np.zeros((W, 3))
    xs = np.clip(x, 0, 1)
    for c in range(3):
        out[:, c] = np.interp(xs, [p for p, _ in STOPS], [col[c] for _, col in STOPS])
    return out

def build_base():
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    g = (238, 240, 243)
    for c in range(COLS + 1):
        dr.line([(c * CW, 0), (c * CW, H)], fill=g)
    for r in range(ROWS + 1):
        dr.line([(0, r * CH), (W, r * CH)], fill=g)
    strip = Image.open(os.path.join(HERE, "icons_strip.png")).convert("RGBA")
    pitch = strip.width / 30 * (300 / 300)
    unit = strip.width / 4456
    icons = [strip.crop((int((i % 15) * 300 * unit), int((i // 15) * 300 * unit), int((i % 15) * 300 * unit + 256 * unit), int((i // 15) * 300 * unit + 256 * unit))) for i in range(30)]
    sz, gap = 50, 13
    for i, ic in enumerate(icons):
        row, col = divmod(i, 15)
        x0 = (W - (15 * sz + 14 * gap)) // 2 + col * (sz + gap)
        y0 = 205 + row * (sz + gap + 4)
        im = ic.resize((sz, sz), Image.LANCZOS)
        img.paste(im, (x0, y0), im)
    card = streak_card()
    if card is not None:
        img.paste(card, ((W - card.width) // 2, 528), card)
    return np.asarray(img, dtype=float)

STREAK_URL = os.environ.get("STREAK_URL", "https://streak-stats.demolab.com/?user=TTNAN&hide_border=true&background=FFFFFF00&ring=FB8C00&fire=FB8C00&currStreakLabel=FB8C00&currStreakNum=1F2328&sideNums=1F2328&sideLabels=1F2328&dates=6E7781&stroke=D0D7DE")
def streak_card():
    import re, urllib.request, io as _io
    try:
        import cairosvg
        svg = urllib.request.urlopen(urllib.request.Request(STREAK_URL, headers={"User-Agent": "koi"}), timeout=30).read().decode()
        svg = re.sub(r"opacity: 0; animation[^']*'", "'", svg)
        svg = re.sub(r"style='animation: currstreak[^']*'", "", svg)
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=680)
        return Image.open(_io.BytesIO(png)).convert("RGBA")
    except Exception as e:
        print("streak card failed:", e)
        return None
BASE = build_base()

def draw_frame(fi, t, energy, cflag, cshade, ckey, crgb, cdim):
    arr = BASE.copy()
    # static ink title + vermilion motto
    arr = arr * (1 - TITLE[..., None]) + np.array([31, 35, 40.0]) * TITLE[..., None]
    arr = arr * (1 - MOTTO[..., None]) + np.array([181, 69, 47.0]) * MOTTO[..., None]

    F = cflag.reshape(ROWS, COLS)
    # soft shadow on the pond floor, offset down-right
    body = (F == FLAG_BODY) | (F == FLAG_RIDGE) | (F == FLAG_EYE)
    shadow = np.zeros_like(body)
    shadow[2:, 2:] = body[:-2, :-2]
    for r, c in zip(*np.nonzero(shadow & (F == 0))):
        arr[r * CH + 1:(r + 1) * CH, c * CW + 1:(c + 1) * CW] *= 0.93
    # translucent fins
    for i in np.nonzero(cflag == FLAG_FIN)[0]:
        c, r = i % COLS, i // COLS
        a = 0.18 + 0.14 * cshade[i]
        reg = arr[r * CH + 1:(r + 1) * CH, c * CW + 1:(c + 1) * CW]
        reg[:] = reg * (1 - a) + crgb[i] * 0.8 * a
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)

    for i in np.nonzero((energy > 0.32) & (cflag == 0))[0]:
        c, r = i % COLS, i // COLS
        a = min(1.0, (energy[i] - 0.32) * 2.2 + 0.18)
        col = tuple(int(255 - (255 - v) * a) for v in (60, 160, 210))
        x = c * CW + CW * 0.66; y = r * CH + CH * 0.64
        dr.rectangle([x - 1, y - 1, x + 1, y + 1], fill=col)

    for i in np.nonzero(cflag)[0]:
        c, r = i % COLS, i // COLS
        x, y = c * CW, r * CH
        f = cflag[i]; sh = cshade[i]; key = int(ckey[i]); base = crgb[i]; dim = cdim[i]
        roll = float(h3(key, fi // 5 + (key % 997) // 97, 7))
        cx, cy = x + CW / 2, y + CH / 2
        if f == FLAG_EDGE:
            st = EDGE_HEAVY if sh >= 0.5 else EDGE_LIGHT
            col = base * (0.42 + 0.2 * (1 - sh))
            dr.text((cx, cy), st[(key >> 8) % len(st)], font=FONT, fill=tuple(int(v) for v in np.clip(col, 0, 255)), anchor="mm")
            continue
        if f == FLAG_FIN:
            col = base * (0.45 + 0.25 * (1 - sh))
            dr.text((cx, cy), FIN_GLYPHS[(key >> 8) % len(FIN_GLYPHS)], font=FONT, fill=tuple(int(v) for v in np.clip(col, 0, 255)), anchor="mm")
            continue
        if f == FLAG_EYE:
            dr.rectangle([x + 1, y + 1, x + CW - 1, y + CH - 1], fill=(20, 20, 24))
            dr.text((cx, cy), "o", font=FONT, fill=(170, 176, 186), anchor="mm")
            continue
        lvl = 0.42 + 0.66 * sh
        if f == FLAG_RIDGE:
            lvl = min(1.08, lvl + 0.1)
        block = np.clip(base * lvl * (0.85 + 0.15 * dim), 0, 255)
        dr.rectangle([x + 1, y + 1, x + CW - 1, y + CH - 1], fill=tuple(int(v) for v in block))
        jit = ((key >> 4) % 3) - 1
        glyph = SKIN_RAMP[int(np.clip(int((1 - sh) * 8.99) + jit, 0, len(SKIN_RAMP) - 1))]
        if f == FLAG_RIDGE:
            glyph = "8%@"[(key >> 8) % 3]
        if roll < 0.06:
            gc = (255, 255, 255)
        elif roll < 0.09:
            gc = tuple(int(v) for v in np.clip(base * 1.25 + 30, 0, 255))
        else:
            gc = tuple(int(v) for v in np.clip(base * 0.32, 0, 255))
        dr.text((cx, cy + 1), glyph, font=FONT, fill=gc, anchor="mm")
    return img

if __name__ == "__main__":
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    energy = np.zeros(COLS * ROWS)
    for fi in range(NF):
        render(fi, energy, draw_img=False)
    if only is not None:
        for fi in range(only):
            render(fi, energy, draw_img=False)
    for fi in ([only] if only is not None else range(NF)):
        render(fi, energy).save(os.path.join(OUT, f"f{fi:04d}.png"))
    print("done", NF, W, H)
