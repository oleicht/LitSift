import html as _html
import re

from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QLabel,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
)

from app.ui.paper import Paper

_SCORE_FIELDS = {
    "rating", "overall_recommendation", "confidence",
    "soundness", "presentation", "contribution",
}

_SKIP_FIELDS = {
    "flag_for_ethics_review", "code_of_conduct", "anonymous_url",
    "no_acknowledgement_section", "submission_guidelines",
}


def _note_type(note):
    for inv in note.invitations:
        inv_lower = inv.lower()
        if "/-/decision" in inv_lower:
            return "decision"
        if "meta" in inv_lower and "review" in inv_lower:
            return "meta_review"
        if "/-/official_review" in inv_lower or inv_lower.endswith("/-/review"):
            return "official_review"
    return "comment"


def _short_name(signature):
    part = signature.split("/")[-1]
    if part.startswith("~"):
        return part[1:].replace("_", " ").rstrip("0123456789").strip()
    return part.replace("_", " ")


def _get(content, key):
    return (content.get(key) or {}).get("value") or ""


def _md(text):
    if not text:
        return ""
    text = _html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    return text.replace("\n", "<br>")


def _body(content):
    for key in ("comment", "rebuttal"):
        val = _get(content, key)
        if val:
            return val
    return ""


def _content_html(content, skip=frozenset()):
    parts = []
    for key, entry in content.items():
        if key in skip:
            continue
        val = (entry or {}).get("value") or ""
        if not isinstance(val, str) or not val.strip():
            continue
        parts.append(f"<b>{key.replace('_', ' ').title()}</b><br>{_md(val)}")
    return "<br>".join(parts)


def _render_thread(note, children_map, depth=0):
    sig = _short_name(note.signatures[0]) if note.signatures else "Unknown"
    title = _get(note.content, "title")
    body = _md(_body(note.content))
    header = f"<b>{_html.escape(sig)}</b>"
    if title:
        header += f" &mdash; <i>{_html.escape(title)}</i>"
    margin = depth * 24
    out = (
        f'<div style="margin-left:{margin}px; border-left:2px solid #ccc;'
        f' padding-left:8px; margin-top:10px;">'
        f"{header}<br>{body}</div>"
    )
    for child in children_map.get(note.id, []):
        out += _render_thread(child, children_map, depth + 1)
    return out


class ReviewDialog(QDialog):
    def __init__(self, paper: Paper, notes: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Reviews — {paper.title}")
        self.resize(900, 700)
        self._build_ui(notes)

    def _build_ui(self, notes):
        layout = QVBoxLayout(self)

        decisions = [n for n in notes if _note_type(n) == "decision"]
        meta = next((n for n in notes if _note_type(n) == "meta_review"), None)
        reviews = [n for n in notes if _note_type(n) == "official_review"]

        children_map: dict[str, list] = {}
        for n in notes:
            if n.replyto:
                children_map.setdefault(n.replyto, []).append(n)
        for children in children_map.values():
            children.sort(key=lambda n: n.cdate or 0)

        for decision in decisions:
            dec = _get(decision.content, "decision")
            label = QLabel(dec)
            font = label.font()
            font.setPointSize(font.pointSize() + 2)
            font.setBold(True)
            label.setFont(font)
            color = "#2a7a2a" if "accept" in dec.lower() else "#aa3333"
            label.setStyleSheet(f"color: {color}; padding: 4px 0;")
            layout.addWidget(label)
            html = _content_html(decision.content, {"decision", "title"} | _SKIP_FIELDS)
            if html:
                browser = QTextBrowser()
                browser.setHtml(html)
                browser.setMaximumHeight(150)
                layout.addWidget(browser)

        if meta:
            group = QGroupBox("Meta Review")
            g_layout = QVBoxLayout(group)
            browser = QTextBrowser()
            browser.setHtml(_content_html(meta.content, _SKIP_FIELDS))
            browser.setMaximumHeight(180)
            g_layout.addWidget(browser)
            layout.addWidget(group)

        if reviews:
            tabs = QTabWidget()
            for review in sorted(reviews, key=lambda n: n.cdate or 0):
                sig = _short_name(review.signatures[0]) if review.signatures else "Reviewer"
                short_id = sig.split()[-1]
                rating = _get(review.content, "rating") or _get(review.content, "overall_recommendation")
                tab_label = f"{short_id} ★{rating}" if rating else short_id
                browser = QTextBrowser()
                browser.setHtml(self._render_review(review, children_map))
                tabs.addTab(browser, tab_label)
            layout.addWidget(tabs)

    def _render_review(self, review, children_map):
        c = review.content
        parts = []

        scores = []
        for key, label in [
            ("rating", "Rating"),
            ("overall_recommendation", "Recommendation"),
            ("confidence", "Confidence"),
            ("soundness", "Soundness"),
            ("presentation", "Presentation"),
            ("contribution", "Contribution"),
        ]:
            val = _get(c, key)
            if val:
                scores.append(f"{label}: <b>{_html.escape(str(val))}</b>")
        if scores:
            parts.append("&nbsp;&nbsp;|&nbsp;&nbsp;".join(scores))
            parts.append("<hr>")

        body = _content_html(c, _SCORE_FIELDS | _SKIP_FIELDS)
        if body:
            parts.append(body)

        thread = children_map.get(review.id, [])
        if thread:
            parts.append("<hr><b>Discussion</b>")
            for note in thread:
                parts.append(_render_thread(note, children_map, depth=0))

        return "<br>".join(parts)
