"""
Frazile AI look: colours, Gradio theme and CSS shared by the Studio and the Admin panel.

Palette: deep navy surfaces with a pink-to-yellow brand gradient. Status colours (OK / warning /
serious / critical) are reserved for state and always come with an icon and a word.
"""

from __future__ import annotations

import inspect

import gradio as gr

PINK = gr.themes.Color(
    name="frazile-pink", c50="#FDF2F8", c100="#FCE7F3", c200="#FBCFE8", c300="#F9A8D4", c400="#F472B6",
    c500="#EC4899", c600="#DB2777", c700="#BE185D", c800="#9D174D", c900="#831843", c950="#500724",
)
YELLOW = gr.themes.Color(
    name="frazile-yellow", c50="#FEFCE8", c100="#FEF9C3", c200="#FEF08A", c300="#FDE047", c400="#FACC15",
    c500="#EAB308", c600="#CA8A04", c700="#A16207", c800="#854D0E", c900="#713F12", c950="#422006",
)
BG, SURFACE, SURFACE_2, BORDER, TEXT, MUTED = "#0B0F1A", "#141A2B", "#0F1422", "#232B40", "#F5F7FF", "#9AA3B8"
ACCENT, ACCENT_2 = "#EC4899", "#FACC15"
GRADIENT = "linear-gradient(90deg, #EC4899 0%, #F59E0B 55%, #FACC15 100%)"
GRADIENT_HOVER = "linear-gradient(90deg, #F472B6 0%, #FBBF24 55%, #FDE047 100%)"
STATUS = {"good": ("#0ca30c", "●", "OK"), "warning": ("#fab219", "▲", "Warm"),
          "serious": ("#ec835a", "▲", "Hot"), "critical": ("#d03b3b", "■", "Critical"),
          "info": ("#3b82f6", "●", "Info"), "neutral": (MUTED, "○", "n/a")}
FAMILY_COLORS = {"sdxl": "#EC4899", "zimage": "#FACC15", "qwen": "#38BDF8", "flux": "#A78BFA",
                 "video": "#34D399", "music": "#FB923C"}


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=PINK,
        secondary_hue=YELLOW,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_lg,
    )
    valid = set(inspect.signature(gr.themes.Base.set).parameters)
    both = {}  # apply every value to light and dark mode so the studio is always dark
    for key, value in {
        "body_background_fill": BG,
        "body_text_color": TEXT,
        "body_text_color_subdued": MUTED,
        "background_fill_primary": SURFACE,
        "background_fill_secondary": SURFACE_2,
        "border_color_primary": BORDER,
        "block_background_fill": SURFACE,
        "block_border_color": BORDER,
        "block_label_background_fill": SURFACE,
        "block_label_text_color": MUTED,
        "block_title_text_color": TEXT,
        "block_info_text_color": MUTED,
        "panel_background_fill": SURFACE,
        "panel_border_color": BORDER,
        "input_background_fill": SURFACE_2,
        "input_background_fill_focus": SURFACE_2,
        "input_border_color": BORDER,
        "input_border_color_focus": ACCENT,
        "input_placeholder_color": "#5B6478",
        "button_primary_background_fill": GRADIENT,
        "button_primary_background_fill_hover": GRADIENT_HOVER,
        "button_primary_text_color": "#1A0A12",
        "button_primary_border_color": "transparent",
        "button_secondary_background_fill": "#1C2436",
        "button_secondary_background_fill_hover": "#273149",
        "button_secondary_text_color": TEXT,
        "button_secondary_border_color": "#2A3350",
        "button_cancel_background_fill": "#3B1F2B",
        "button_cancel_background_fill_hover": "#4C2435",
        "button_cancel_text_color": "#FCA5A5",
        "button_cancel_border_color": "#5B2A3A",
        "table_even_background_fill": "#121829",
        "table_odd_background_fill": SURFACE_2,
        "table_border_color": BORDER,
        "table_row_focus": "#2A1A2E",
        "checkbox_background_color": SURFACE_2,
        "checkbox_background_color_selected": ACCENT,
        "checkbox_border_color": "#2A3350",
        "checkbox_label_background_fill": SURFACE_2,
        "checkbox_label_background_fill_selected": "#2A1A2E",
        "checkbox_label_text_color": TEXT,
        "checkbox_label_border_color": "#2A3350",
        "color_accent": ACCENT,
        "color_accent_soft": "#2A1A2E",
        "link_text_color": ACCENT_2,
        "link_text_color_hover": "#FDE047",
        "slider_color": ACCENT,
        "accordion_text_color": TEXT,
        "shadow_drop": "0 1px 2px rgba(0,0,0,.35)",
        "shadow_drop_lg": "0 12px 32px -12px rgba(236,72,153,.35)",
        "block_shadow": "none",
    }.items():
        for k in (key, f"{key}_dark"):
            if k in valid:
                both[k] = value
    return theme.set(**both)


CSS = f"""
.gradio-container {{ max-width: none !important; padding: 0 28px 28px !important; }}
footer {{ display: none !important; }}

/* header */
.fz-header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px;
              padding: 14px 4px 14px; border-bottom: 1px solid {BORDER}; margin-bottom: 12px; }}
.fz-brand {{ display: flex; align-items: center; gap: 12px; }}
.fz-mark {{ width: 42px; height: 42px; border-radius: 12px; background: {GRADIENT};
            box-shadow: 0 8px 24px -8px rgba(236,72,153,.7); display: grid; place-items: center;
            font-weight: 900; font-size: 21px; color: #1A0A12; letter-spacing: -.04em; }}
.fz-logo {{ font-weight: 800; font-size: 26px; letter-spacing: -.02em; line-height: 1; color: {TEXT}; }}
.fz-logo span {{ background: {GRADIENT}; -webkit-background-clip: text; background-clip: text; color: transparent; }}
.fz-tag {{ color: {MUTED}; font-size: 13px; margin-top: 5px; }}
.fz-sys {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
.fz-pill {{ font-family: var(--font-mono); font-size: 11.5px; color: {MUTED}; border: 1px solid {BORDER};
            background: {SURFACE}; border-radius: 999px; padding: 5px 10px; white-space: nowrap; }}
.fz-pill b {{ color: {ACCENT_2}; font-weight: 600; }}
.fz-pill.live {{ border-color: #3A2C4A; }}
.fz-pill .dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }}

/* generate tab */
.fz-interp {{ border-left: 3px solid {ACCENT}; background: #17122A; border-radius: 10px;
              padding: 8px 12px; font-size: 13px; margin: 2px 0 8px; min-height: 20px; }}
.fz-interp p {{ margin: 0; }}
.fz-interp small {{ color: {MUTED}; }}
.fz-loaded {{ font-size: 12px; color: {MUTED}; margin: -4px 0 4px 2px; }}
.fz-loaded b {{ color: #F9A8D4; font-weight: 600; }}
.fz-chips {{ gap: 6px !important; flex-wrap: wrap; }}
.fz-chips button {{ font-size: 12px !important; padding: 4px 10px !important; min-width: 0 !important;
                   border-radius: 999px !important; height: 30px !important; flex: 0 0 auto !important; }}
.fz-chips-label {{ font-size: 11.5px; color: {MUTED}; margin: 2px 0 -2px 2px; text-transform: uppercase; letter-spacing: .06em; }}
.fz-upscale {{ border: 1px dashed #3A2C4A; border-radius: 14px; padding: 10px 12px;
               background: linear-gradient(180deg, #151225 0%, {SURFACE} 100%); margin-top: 8px; }}
.fz-hint {{ color: {MUTED}; font-size: 12px; }}
.fz-empty {{ border: 1px dashed {BORDER}; border-radius: 16px; padding: 40px 24px; text-align: center; color: {MUTED};
             background: radial-gradient(600px 200px at 50% 0%, rgba(236,72,153,.08), transparent 70%); }}
.fz-empty b {{ color: {TEXT}; font-size: 16px; display: block; margin-bottom: 6px; }}
.fz-empty code {{ color: #FDE68A; }}

button.primary {{ font-weight: 700; letter-spacing: .01em;
                  box-shadow: 0 10px 28px -10px rgba(236,72,153,.75); transition: transform .08s ease; }}
button.primary:hover {{ transform: translateY(-1px); }}
button.secondary {{ font-weight: 600; }}

.tabs > .tab-nav button.selected, .tab-nav button.selected {{ color: {ACCENT_2} !important; border-color: {ACCENT} !important; }}
.tab-nav button {{ font-weight: 600; }}
.prose h3 {{ color: {ACCENT_2}; }}
.prose h4 {{ color: #F9A8D4; margin-top: 4px; }}
.prose table th {{ color: #F9A8D4; }}
.prose code {{ background: #1C1530; color: #FDE68A; border: 1px solid #3A2C4A; }}
.gallery-item, .thumbnail-item {{ border-radius: 12px !important; }}
.progress-bar {{ background: {GRADIENT} !important; }}

/* cards, badges, tables */
.fz-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 10px; margin: 4px 0 8px; }}
.fz-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 12px 14px; position: relative; }}
.fz-card.active {{ border-color: {ACCENT}; box-shadow: 0 0 0 1px {ACCENT} inset; }}
.fz-card .t {{ font-weight: 700; font-size: 14px; color: {TEXT}; margin-right: 60px; }}
.fz-card .d {{ font-size: 12px; color: {MUTED}; margin-top: 4px; line-height: 1.4; }}
.fz-card .m {{ font-family: var(--font-mono); font-size: 11px; color: {MUTED}; margin-top: 8px; }}
.fz-card .m b {{ color: {TEXT}; font-weight: 600; }}
.fz-card .corner {{ position: absolute; top: 10px; right: 12px; display: flex; gap: 4px; }}
.fz-badge {{ display: inline-block; font-size: 10.5px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
             padding: 2px 7px; border-radius: 999px; border: 1px solid transparent; line-height: 1.5; }}
.fz-badge.ok {{ color: #86efac; background: rgba(12,163,12,.15); border-color: rgba(12,163,12,.35); }}
.fz-badge.off {{ color: {MUTED}; background: #1A2033; border-color: {BORDER}; }}
.fz-badge.warn {{ color: #fde68a; background: rgba(250,178,25,.15); border-color: rgba(250,178,25,.35); }}
.fz-badge.bad {{ color: #fca5a5; background: rgba(208,59,59,.15); border-color: rgba(208,59,59,.35); }}
.fz-badge.fam {{ color: #1A0A12; }}
.fz-note {{ font-size: 12.5px; color: {MUTED}; border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 12px; background: {SURFACE_2}; }}
.fz-note b {{ color: {TEXT}; }}
.fz-ok {{ color: #86efac; }}  .fz-bad {{ color: #fca5a5; }}  .fz-warn {{ color: #fde68a; }}

/* admin panel */
.fz-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 10px; margin: 4px 0 12px; }}
.fz-tile {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 12px 14px 10px; min-height: 92px; }}
.fz-tile .l {{ font-size: 12px; color: {MUTED}; }}
.fz-tile .v {{ font-size: 24px; font-weight: 600; color: {TEXT}; line-height: 1.15; margin-top: 4px; }}
.fz-tile .v small {{ font-size: 13px; color: {MUTED}; font-weight: 500; margin-left: 2px; }}
.fz-tile .s {{ font-size: 11.5px; color: {MUTED}; margin-top: 4px; }}
.fz-tile .st {{ font-size: 11.5px; font-weight: 600; margin-top: 4px; }}
.fz-meter {{ height: 6px; border-radius: 3px; background: #3B2238; margin-top: 8px; overflow: hidden; }}
.fz-meter > div {{ height: 100%; border-radius: 3px; background: {ACCENT}; }}
.fz-cores {{ display: flex; gap: 3px; margin-top: 8px; align-items: flex-end; height: 22px; }}
.fz-cores > div {{ flex: 1; background: #3B2238; border-radius: 2px; position: relative; height: 100%; }}
.fz-cores > div > div {{ position: absolute; bottom: 0; left: 0; right: 0; background: {ACCENT}; border-radius: 2px; }}
.fz-admin-head {{ display: flex; justify-content: space-between; align-items: center; color: {MUTED}; font-size: 12px; }}
.fz-jobs {{ display: flex; flex-direction: column; gap: 8px; }}
.fz-job {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 10px 14px; }}
.fz-job .r {{ display: flex; justify-content: space-between; gap: 12px; font-size: 13px; }}
.fz-job .r b {{ color: {TEXT}; }}
.fz-job .r span {{ color: {MUTED}; font-family: var(--font-mono); font-size: 11.5px; }}
.fz-job .fz-meter {{ margin-top: 6px; }}
.fz-job .msg {{ font-size: 12px; color: {MUTED}; margin-top: 5px; }}
.fz-log textarea {{ font-family: var(--font-mono) !important; font-size: 11.5px !important; line-height: 1.5 !important;
                    background: #0A0D16 !important; color: #C9D1E3 !important; }}
.fz-section {{ border: 1px solid {BORDER}; border-radius: 14px; padding: 10px 14px 6px; background: {SURFACE}; }}
.fz-kv table {{ width: 100%; }}
.fz-kv td:first-child {{ color: {MUTED}; white-space: nowrap; width: 200px; }}

/* responsive layout and interaction polish */
.gradio-container, .gradio-container * {{ box-sizing: border-box; }}
.gradio-container {{ min-width: 0; padding-inline: clamp(12px, 2.5vw, 28px) !important; }}
.gradio-container .row > .column, .gradio-container .row > .block {{ min-width: 0; }}
.gradio-container .tabs > .tab-nav {{ max-width: 100%; overflow-x: auto; flex-wrap: nowrap; scrollbar-width: thin; }}
.gradio-container .tab-nav button {{ flex: 0 0 auto; white-space: nowrap; }}
.gradio-container button:focus-visible, .gradio-container a:focus-visible {{ outline: 2px solid {ACCENT_2}; outline-offset: 2px; }}
.fz-header {{ flex-wrap: wrap; }}
.fz-sys {{ min-width: 0; }}
.fz-empty {{ padding: clamp(20px, 4vw, 40px) clamp(14px, 3vw, 24px); }}
.fz-kv td:first-child {{ width: min(40%, 200px); white-space: normal; overflow-wrap: anywhere; }}
.fz-admin-head {{ flex-wrap: wrap; gap: 4px 14px; }}
.gradio-container [data-testid="dataframe"] {{ max-width: 100%; overflow-x: auto; }}
.gradio-container .gallery .grid-wrap {{ grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)) !important; }}
.gradio-container .gallery {{ height: min(640px, 66vh) !important; min-height: 280px; }}

@media (max-width: 900px) {{
  .gradio-container {{ padding-bottom: 20px !important; }}
  .gradio-container .row {{ flex-wrap: wrap; }}
  .gradio-container .row > .column {{ flex: 1 1 100% !important; min-width: 0 !important; }}
  .fz-header {{ align-items: flex-start; }}
  .fz-sys {{ justify-content: flex-start; }}
  .fz-kv td:first-child {{ width: 36%; }}
}}

@media (max-width: 640px) {{
  .gradio-container {{ padding-inline: 12px !important; }}
  .fz-header {{ gap: 10px; padding: 12px 0; }}
  .fz-brand {{ gap: 9px; }}
  .fz-mark {{ width: 36px; height: 36px; border-radius: 10px; font-size: 18px; }}
  .fz-logo {{ font-size: 22px; }}
  .fz-tag {{ font-size: 12px; }}
  .fz-sys {{ justify-content: flex-start; gap: 6px; }}
  .fz-pill {{ font-size: 10.5px; padding: 5px 8px; }}
  .gradio-container .row:not(.fz-chips) > .block {{ flex: 1 1 calc(50% - 8px) !important; min-width: 0 !important; }}
  .gradio-container .row.fz-upscale > .block {{ flex: 1 1 100% !important; }}
  .gradio-container button {{ min-height: 40px; }}
  .fz-chips button {{ min-height: 38px !important; height: auto !important; padding: 6px 10px !important; }}
  .gradio-container .gallery {{ height: min(420px, 52vh) !important; min-height: 220px; }}
  .gradio-container .gallery .grid-wrap {{ grid-template-columns: repeat(auto-fill, minmax(125px, 1fr)) !important; }}
  .fz-tile {{ min-height: 84px; padding: 10px 11px; }}
  .fz-tile .v {{ font-size: 21px; }}
  .fz-job {{ padding: 10px; }}
  .fz-job .r {{ flex-wrap: wrap; }}
  .fz-kv td:first-child {{ width: 34%; }}
}}

@media (max-width: 420px) {{
  .gradio-container .row:not(.fz-chips):not(.fz-upscale) > .block {{ flex-basis: 100% !important; }}
  .fz-logo {{ font-size: 20px; }}
  .fz-pill {{ white-space: normal; }}
}}

@media (prefers-reduced-motion: reduce) {{
  .gradio-container *, .gradio-container *::before, .gradio-container *::after {{ scroll-behavior: auto !important; transition-duration: 0.01ms !important; }}
}}
"""


def badge(text: str, kind: str = "off") -> str:
    return f'<span class="fz-badge {kind}">{text}</span>'


def family_badge(family: str) -> str:
    color = FAMILY_COLORS.get(family, MUTED)
    return f'<span class="fz-badge fam" style="background:{color}">{family}</span>'


def status_span(status: str, text: str | None = None) -> str:
    color, icon, word = STATUS.get(status, STATUS["neutral"])
    return f'<span style="color:{color};font-weight:600">{icon} {text or word}</span>'
