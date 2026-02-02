# 📐 Figure Specifications for External Recreation

These 4 figures need to be recreated in professional design software (draw.io, Figma, or PowerPoint) for publication quality.

---

## 1. System Flowchart (`system_flowchart.png`)

### Purpose
Shows how ECM Battery, Thermal, 5G Network, and Load models interconnect.

### Dimensions
- **Canvas**: 1400 × 900 px (or proportional)
- **Export**: PNG at 300 DPI

### Layout (5 boxes arranged in 2×2 + 1)

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│    ┌──────────────┐          I (current)         ┌──────────────┐  │
│    │              │ ─────────────────────────────▶│              │  │
│    │  ECM Battery │                               │   5G RRC     │  │
│    │              │                               │              │  │
│    └──────────────┘                               └──────────────┘  │
│           │ ▲                                            │          │
│       T   │ │ I²R                                   Pnet │          │
│           ▼ │                                            ▼          │
│    ┌──────────────┐          Qheat               ┌──────────────┐  │
│    │              │ ◀─────────────────────────────│              │──▶ [Output]
│    │   Thermal    │                               │  Load Model  │  │
│    │              │                               │              │  │
│    └──────────────┘                               └──────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Box Specifications

| Box | Color (Fill) | Color (Border) | Width | Height |
|-----|--------------|----------------|-------|--------|
| ECM Battery | `#3498DB` 20% opacity | `#3498DB` solid 3px | 200px | 140px |
| Thermal Model | `#E74C3C` 20% opacity | `#E74C3C` solid 3px | 200px | 140px |
| 5G RRC | `#9B59B6` 20% opacity | `#9B59B6` solid 3px | 200px | 140px |
| Load Model | `#27AE60` 20% opacity | `#27AE60` solid 3px | 200px | 140px |
| Output | `#F39C12` 20% opacity | `#F39C12` solid 3px | 140px | 100px |

### Box Content (Use bold sans-serif, 12pt)

**ECM Battery:**
```
ECM Battery
───────────
dSOC/dt = −I/Q
dV/dt = (I−V/R)/C
```

**Thermal Model:**
```
Thermal Model
───────────
M·dT/dt = I²R − hA(T−Tamb)
```

**5G RRC:**
```
5G RRC
───────────
IDLE → CONNECTED → TAIL
P = f(state)
```

**Load Model:**
```
Load Model
───────────
OLED + CPU + GPU + Radio
P = ΣPi
```

**Output:**
```
Output
───────────
SOC, V, T, Life
```

### Arrow Specifications

- **Color**: Dark gray `#2C3E50`
- **Width**: 3px
- **Style**: Solid with filled arrowhead

| From | To | Label | Position |
|------|----|-------|----------|
| Load Model | ECM Battery | **I (current)** | Above arrow |
| ECM Battery | Thermal | **T** (down), **I²R** (up) | Beside bidirectional arrow |
| 5G RRC | Load Model | **Pnet** | Right of arrow |
| Load Model | Thermal | **Qheat** | Above arrow |
| Load Model | Output | *(no label)* | Simple arrow |

---

## 2. ECM Circuit Diagram (`ecm_circuit_diagram.png`)

### Purpose
Shows equivalent circuit: OCV source → R₀ → RC₁ parallel → RC₂ parallel → Terminal

### Dimensions
- **Canvas**: 1200 × 600 px
- **Export**: PNG at 300 DPI

### Circuit Layout

```
                    ┌──── R₁ ────┐       ┌──── R₂ ────┐
                    │            │       │            │
   ⊕                │            │       │            │           I
  ─┼─ ────▬▬▬▬──────┼────────────┼───────┼────────────┼────────●────▶ U
  OCV      R₀       │            │       │            │       Terminal
                    │    ┤├      │       │    ┤├      │
                    └─────┴──────┘       └─────┴──────┘
                          C₁                   C₂
```

### Color Scheme

| Component | Color | Hex Code |
|-----------|-------|----------|
| OCV Source | Blue | `#2E86AB` |
| R₀ (ohmic) | Red | `#E94F37` |
| RC₁ pair | Green | `#44AF69` |
| RC₂ pair | Orange | `#F18F01` |
| Wires | Dark Gray | `#333333` |

### Component Labels

| Component | Main Label | Sublabel |
|-----------|------------|----------|
| OCV | **OCV(SOC, T)** | - |
| R₀ | **R₀(SOC, T)** | *Instantaneous* |
| R₁, C₁ | **R₁**, **C₁** | **τ₁ ≈ 36s** *(Activation)* |
| R₂, C₂ | **R₂**, **C₂** | **τ₂ ≈ 300s** *(Concentration)* |

### Master Equation (Bottom, in gray box `#F5F5F5`)

```
U = OCV(SOC, T) − I·R₀(SOC, T) − ΔU_RC1 − ΔU_RC2
```

---

## 3. KiBaM Schematic (`figure1_kibam_schematic.png`)

### Purpose
Two-tank battery analogy with available and bound charge.

### Dimensions
- **Canvas**: 900 × 700 px
- **Export**: PNG at 300 DPI

### Layout

```
     Available Charge              Bound Charge
   q₁ (Direct Power)            q₂ (Chemical Reserve)

   ┌───────────────┐            ┌───────────────┐
   │███████████████│◀──────────▶│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
   │███████████████│  Recovery  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
   │               │    Flow    │               │
   └───────┬───────┘            └───────────────┘
           │
           ▼
    Load Current I(t)
```

### Tank Colors

| Tank | Fill | Border |
|------|------|--------|
| Available (q₁) | Light Blue `#AED6F1` | Blue `#2980B9` 3px |
| Bound (q₂) | Light Orange `#FAD7A0` | Orange `#E67E22` 3px |
| Fluid in q₁ | `#3498DB` | - |
| Fluid in q₂ | `#F39C12` | - |

### Recovery Flow Label (Green `#27AE60`, between tanks)
```
Recovery Flow: k(q₂ − q₁·(1−c)/c)
```

### Equations Box (Light yellow `#FFF9E6`)
```
dq₁/dt = −I(t) + k(q₂ − q₁·(1−c)/c)
dq₂/dt = −k(q₂ − q₁·(1−c)/c)
```

---

## 4. Thermal Feedback Loop (`ecm_thermal_feedback.png`)

### Purpose
Circular feedback: Current → Heat → Temp → Resistance → Current

### Dimensions
- **Canvas**: 600 × 500 px
- **Export**: PNG at 300 DPI

### Layout (Circular)

```
              ┌───────────┐
              │ Current   │
              │   I(t)    │
              └─────┬─────┘
         ▲          │
         │          ▼
    ┌────┴────┐    ┌───────────┐
    │  R₀(T)  │    │ Heat Gen  │
    │   ↑     │    │  Q = I²R  │
    └────┬────┘    └─────┬─────┘
         ▲               │
         │               ▼
         │         ┌───────────┐
         │         │ Temp Rise │
         └─────────┤   dT/dt   │
                   └───────────┘
```

### Box Colors

| Box | Fill | Border |
|-----|------|--------|
| Current I(t) | `#D6EAF8` | `#2980B9` 2px |
| Heat Gen Q=I²R | `#FADBD8` | `#C0392B` 2px |
| Temp Rise dT/dt | `#FCF3CF` | `#E67E22` 2px |
| R₀(T) ↑ | `#D5F5E3` | `#229954` 2px |

### Feedback Arrow
- **Color**: Red `#E74C3C`, 2px, **dashed**
- **Label**: "Feedback" (red, italic)

### Optional Callout (yellow box `#FFF9E6`)
```
⚠️ Positive Feedback Loop:
Higher I → More Heat → Higher T → Higher R → Even MORE Heat
```

---

## 🎨 Recommended Tools

| Tool | Best For | Cost |
|------|----------|------|
| **draw.io** | System flowchart, Thermal feedback | Free |
| **Figma** | KiBaM schematic, ECM circuit | Free tier |
| **Lucidchart** | All diagrams | Free tier |

### draw.io Tips
1. Go to [diagrams.net](https://app.diagrams.net/)
2. For circuit symbols: Insert → Advanced → Electrical
3. Export: File → Export as → PNG (300% scale)

---

## 📁 Save To

```
figures/system_flowchart.png
figures/ecm_circuit_diagram.png
figures/figure1_kibam_schematic.png
figures/ecm_thermal_feedback.png
```

---

## ✅ Checklist

- [ ] Resolution 300 DPI+
- [ ] Text readable (min 10pt)
- [ ] Colors match hex codes
- [ ] No overlapping text
- [ ] Clear arrow directions
- [ ] White background
- [ ] Consistent font (Arial/Helvetica)
