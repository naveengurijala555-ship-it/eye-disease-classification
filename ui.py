"""Look and feel for the app: CSS and small HTML builders (no TensorFlow or Streamlit needed)."""
from html import escape

CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
LABELS = {
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "normal": "Normal",
}
INFO = {
    "cataract": "Cataract is a clouding of the eye's lens that can blur vision.",
    "diabetic_retinopathy": "Diabetic retinopathy is damage to the retina's blood vessels caused by diabetes.",
    "glaucoma": "Glaucoma is damage to the optic nerve, often linked to raised eye pressure.",
    "normal": "The model did not find signs of the three diseases in this image.",
}

CSS = """<style>
.block-container { max-width: 940px; padding-top: 1.5rem; padding-bottom: 3rem; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.hero { background: #9E1B32; color: #fff; border-radius: 16px; padding: 28px 30px; margin-bottom: 18px; }
.hero h1 { font-family: Cambria, Georgia, serif; font-size: 1.75rem; line-height: 1.25; margin: 0 0 10px 0; color: #fff; padding: 0; }
.hero p { margin: 0; font-size: 0.95rem; opacity: 0.92; color: #fff; }
.hero .tag { display: inline-block; font-size: 0.72rem; letter-spacing: 0.14em; font-weight: 700; background: rgba(255,255,255,0.18); border-radius: 999px; padding: 4px 12px; margin-bottom: 12px; }
.note { background: #F6EEF0; border-radius: 12px; padding: 12px 16px; font-size: 0.92rem; color: #3E4C59; margin-bottom: 14px; }
.empty { border: 2px dashed #D9CDD0; border-radius: 14px; padding: 22px 24px; margin-top: 12px; color: #3E4C59; }
.empty h4 { margin: 0 0 8px 0; color: #9E1B32; font-family: Cambria, Georgia, serif; }
.empty ul { margin: 0; padding-left: 20px; }
.empty li { margin: 4px 0; }
.result { background: #F6EEF0; border-radius: 16px; padding: 22px 24px; }
.result .kicker { font-size: 0.72rem; letter-spacing: 0.14em; font-weight: 700; color: #52606D; }
.result .name { font-family: Cambria, Georgia, serif; font-size: 1.9rem; font-weight: 700; color: #9E1B32; margin: 4px 0 8px 0; line-height: 1.2; }
.result .name.unsure { color: #52606D; }
.result .pill { display: inline-block; background: #9E1B32; color: #fff; font-weight: 700; font-size: 0.9rem; border-radius: 999px; padding: 4px 14px; }
.result .pill.unsure { background: #7B8794; }
.result .info { margin: 12px 0 0 0; font-size: 0.93rem; color: #3E4C59; }
.bars { margin-top: 18px; }
.bar-row { margin: 10px 0; }
.bar-label { display: flex; justify-content: space-between; font-size: 0.88rem; color: #1F2933; margin-bottom: 4px; }
.bar-track { background: #fff; border-radius: 999px; height: 10px; overflow: hidden; }
.bar-fill { height: 10px; border-radius: 999px; background: #C9B8BD; }
.bar-fill.top { background: #9E1B32; }
.fine { font-size: 0.8rem; color: #52606D; margin-top: 14px; }
table.res { border-collapse: collapse; width: 100%; font-size: 0.92rem; margin: 6px 0 14px 0; }
table.res th { background: #9E1B32; color: #fff; padding: 8px 10px; text-align: center; font-weight: 600; }
table.res td { border: 1px solid #E4D9DC; padding: 8px 10px; text-align: center; color: #1F2933; }
table.res th:first-child, table.res td:first-child { text-align: left; }
.about h3 { color: #9E1B32; font-family: Cambria, Georgia, serif; margin-top: 22px; }
.about li { margin: 4px 0; }
</style>"""


def hero_html():
    return (
        '<div class="hero"><div class="tag">AI &amp; DATA SCIENCE MINI PROJECT</div>'
        "<h1>Deep Learning-Based Multi-Class Eye Disease Classification using CNN</h1>"
        "<p>Upload an eye image and a fine-tuned MobileNetV2 CNN predicts Normal, Diabetic Retinopathy, Cataract or Glaucoma.</p></div>"
    )


def note_html():
    return '<div class="note"><b>Screening aid only.</b> This is a college project and not a medical diagnosis. Please consult an eye doctor for any health concern.</div>'


def empty_html():
    return (
        '<div class="empty"><h4>How to use</h4><ul>'
        "<li>Click <b>Browse files</b> above and choose a JPG or PNG image.</li>"
        "<li>Use a clear, well-lit retina (fundus) photo for the best result.</li>"
        "<li>The prediction and confidence appear here in a few seconds.</li>"
        "</ul></div>"
    )


def result_html(probs, threshold=0.6):
    best = max(range(len(probs)), key=lambda i: probs[i])
    conf = float(probs[best])
    sure = conf >= threshold
    key = CLASS_NAMES[best]
    if sure:
        head = (
            '<div class="kicker">PREDICTION</div>'
            f'<div class="name">{escape(LABELS[key])}</div>'
            f'<span class="pill">{conf * 100:.1f}% confidence</span>'
            f'<p class="info">{escape(INFO[key])}</p>'
        )
    else:
        head = (
            '<div class="kicker">PREDICTION</div>'
            '<div class="name unsure">Uncertain</div>'
            f'<span class="pill unsure">{conf * 100:.1f}% top confidence</span>'
            '<p class="info">The model is not confident. Please upload a clear fundus image.</p>'
        )
    rows = ""
    for i in sorted(range(len(probs)), key=lambda i: -probs[i]):
        pct = float(probs[i]) * 100
        cls = "bar-fill top" if (i == best and sure) else "bar-fill"
        rows += (
            f'<div class="bar-row"><div class="bar-label"><span>{escape(LABELS[CLASS_NAMES[i]])}</span><span>{pct:.1f}%</span></div>'
            f'<div class="bar-track"><div class="{cls}" style="width:{pct:.1f}%"></div></div></div>'
        )
    return (
        f'<div class="result">{head}<div class="bars">{rows}</div>'
        '<div class="fine">Confidence is the model\'s own score, not a medical probability.</div></div>'
    )


def about_html():
    return (
        '<div class="about">'
        "<h3>About the model</h3>"
        "<ul>"
        "<li>Fine-tuned <b>MobileNetV2</b> (a lightweight CNN) trained on 5,200 images in 4 classes.</li>"
        "<li>Tested on 446 unseen images: about <b>84.5% accuracy</b> and macro F1 of <b>0.86</b>.</li>"
        "<li>A custom CNN built from scratch reached about 72% validation accuracy, so transfer learning helped clearly.</li>"
        "</ul>"
        "<h3>Test results by class</h3>"
        '<table class="res"><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr>'
        "<tr><td>Cataract</td><td>1.00</td><td>1.00</td><td>1.00</td></tr>"
        "<tr><td>Diabetic retinopathy</td><td>0.95</td><td>0.94</td><td>0.94</td></tr>"
        "<tr><td>Glaucoma</td><td>0.80</td><td>0.71</td><td>0.76</td></tr>"
        "<tr><td>Normal</td><td>0.69</td><td>0.80</td><td>0.74</td></tr></table>"
        "<h3>Limitations</h3>"
        "<ul>"
        "<li>Glaucoma and normal images are often confused.</li>"
        "<li>Cataract images are outer-eye photos while the other classes are retina photos, so cataract is unusually easy for the model.</li>"
        "<li>Not tested on other cameras or datasets, so results on new images may be lower.</li>"
        "</ul></div>"
    )


def team_html():
    return (
        '<div class="about"><h3>Team</h3>'
        "<p>G. Naveen, G. Chakravarthy, S. Bhargav<br>"
        "Guide: K. Lakshmi Padmavathi, Sr. Assistant Professor<br>"
        "Dept. of Artificial Intelligence &amp; Data Science, Lakireddy Bali Reddy College of Engineering (Autonomous), Mylavaram</p></div>"
    )
