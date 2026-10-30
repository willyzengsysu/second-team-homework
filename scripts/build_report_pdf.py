"""把 S7 同源 Markdown 排成 9 页课程报告；需 requirements-report.txt。"""
from __future__ import annotations

import html
import re
import hashlib
import io
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser, math_to_image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph as BaseParagraph, Spacer, Table,
                               TableStyle, PageBreak, Image, Flowable)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 100
MATH_PARSER = MathTextParser("path")
MATH_CACHE = ROOT / "tmp/pdfs/math"


class Paragraph(BaseParagraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ReportLab 的 CJK 换行器会对图像的空 text 调用 ord()。
        # 对象替代符使图像作为一个不可拆分字符参与换行；实际绘制仍由 cbDefn 执行。
        for frag in self.frags:
            if not frag.text and getattr(getattr(frag, "cbDefn", None), "kind", None) == "img":
                frag.text = "\ufffc"


@lru_cache(maxsize=256)
def math_layout(expression, size):
    """仅解析数学表达式，不调用外部 TeX 或 shell。"""
    prop = FontProperties(size=size, math_fontfamily="stix")
    return MATH_PARSER.parse(f"${expression}$", dpi=72, prop=prop)


class MathFormula(Flowable):
    """以嵌入字体和矢量分数线绘制独立公式。"""
    def __init__(self, expression, size=11.5):
        super().__init__()
        self.layout = math_layout(expression, size)
        self.width, self.height = float(self.layout.width), float(self.layout.height) + 4
        self.hAlign = "CENTER"
        self.spaceBefore, self.spaceAfter = 3, 7

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#24313B"))
        offset = float(self.layout.depth) + 2
        for glyph in self.layout.glyphs:
            # Matplotlib 3.11 增加 glyph index；字符码和末两项坐标保持不变。
            font, size, codepoint = glyph[:3]
            x, y = glyph[-2:]
            name = "Math-" + hashlib.sha256(str(font.fname).encode()).hexdigest()[:12]
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, font.fname))
            canvas.setFont(name, size)
            canvas.drawString(float(x), float(y) + offset, chr(codepoint))
        for x, y, width, height in self.layout.rects:
            canvas.rect(float(x), float(y) + offset, float(width), float(height), fill=1, stroke=0)
        canvas.restoreState()


@lru_cache(maxsize=256)
def inline_math(expression, size):
    """行内公式使用透明 400 dpi 图像，按数学基线与正文对齐。"""
    layout = math_layout(expression, size)
    MATH_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{size}:{expression}".encode()).hexdigest()
    path = MATH_CACHE / f"{key}.png"
    with matplotlib.rc_context({"savefig.transparent": True}):
        math_to_image(f"${expression}$", path,
                      prop=FontProperties(size=size, math_fontfamily="stix"),
                      dpi=400, format="png", color="#24313B")
    return (f'<img src="{path.as_posix()}" width="{float(layout.width):.3f}" '
            f'height="{float(layout.height):.3f}" valign="{-float(layout.depth):.3f}"/>')


def plain_inline(text):
    text = html.escape(text)
    text = text.replace("R²", "R<super>2</super>")
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#1F5575">\1</link>', text)
    return text


def inline(text, size=10.4):
    parts = re.split(r"(\$[^$\n]+\$)", text)
    return "".join(inline_math(part[1:-1], size) if part.startswith("$")
                   and part.endswith("$") else plain_inline(part) for part in parts)


def main():
    # Windows 项目沿用系统中文字体。换系统时可把等效字体放在 report/fonts/。
    font = Path("C:/Windows/Fonts/simsun.ttc")
    bold = Path("C:/Windows/Fonts/simhei.ttf")
    if not font.exists():
        font = ROOT / "report/fonts/NotoSansSC-Regular.ttf"
        bold = ROOT / "report/fonts/NotoSansSC-Bold.ttf"
    pdfmetrics.registerFont(TTFont("CJK", str(font), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CJK-Bold", str(bold)))
    pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJK-Bold", italic="CJK", boldItalic="CJK-Bold")
    styles = {
        "body": ParagraphStyle("body", fontName="CJK", fontSize=10.4, leading=16.5,
                               wordWrap="CJK", spaceAfter=7, textColor=colors.HexColor("#24313B")),
        "title": ParagraphStyle("title", fontName="CJK-Bold", fontSize=20, leading=27,
                                spaceBefore=5, spaceAfter=16, textColor=colors.HexColor("#173E52")),
        "h1": ParagraphStyle("h1", fontName="CJK-Bold", fontSize=16, leading=22, spaceAfter=14,
                             textColor=colors.HexColor("#173E52")),
        "h2": ParagraphStyle("h2", fontName="CJK-Bold", fontSize=11.8, leading=18,
                             spaceBefore=7, spaceAfter=6, keepWithNext=True, textColor=colors.HexColor("#173E52")),
        "cell": ParagraphStyle("cell", fontName="CJK", fontSize=8.3, leading=12.5, wordWrap="CJK"),
        "head": ParagraphStyle("head", fontName="CJK-Bold", fontSize=8.3, leading=12.5,
                               wordWrap="CJK", textColor=colors.white),
    }
    lines = (ROOT / "report/报告.md").read_text(encoding="utf-8").splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- PAGEBREAK -->":
            story.append(PageBreak())
            i += 1
            continue
        if line == "$$":
            expression = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                expression.append(lines[i].strip())
                i += 1
            if i == len(lines):
                raise ValueError("独立公式缺少结束 $$")
            formula = MathFormula(" ".join(expression))
            if formula.width > WIDTH:
                raise ValueError(f"公式超出正文宽度：{expression}")
            story.append(formula)
            i += 1
            continue
        if line.startswith("# "):
            style = "title" if not story else "h1"
            story.append(Paragraph(inline(line[2:]), styles[style]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["h2"]))
        elif line.startswith("!["):
            match = re.match(r"!\[.*?\]\((.*?)\)", line)
            path = (ROOT / "report" / match.group(1)).resolve()
            image = Image(str(path))
            scale = min(WIDTH / image.imageWidth, 190 / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
            image.hAlign = "CENTER"
            story.extend([image, Spacer(1, 7)])
        elif line.startswith("|"):
            raw = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in row):
                    raw.append(row)
                i += 1
            n = len(raw[0])
            if n == 2:
                shares = [.27,.73] if raw[0][0] == "数据" else [.65,.35]
            elif n == 3:
                shares = [.20,.62,.18] if raw[0][0] == "阶段" else [.20,.36,.44]
            elif n == 4:
                shares = [.49,.17,.17,.17] if raw[0][0] == "情景" else [.40,.15,.225,.225]
            elif n == 5:
                shares = [.14,.22,.23,.23,.18]
            else:
                shares = [1/n] * n
            data = [[Paragraph(inline(cell, 8.3), styles["head" if j==0 else "cell"])
                     for cell in row] for j,row in enumerate(raw)]
            tbl = Table(data, colWidths=[WIDTH*x for x in shares], repeatRows=1, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244C61")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F1F5F7"),colors.white]),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6),
                ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#BBCAD2")),
            ]))
            story.extend([tbl, Spacer(1,9)])
            continue
        else:
            paragraph = [line]
            while i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].lstrip().startswith(("#", "|", "![", "<!--", "$$")):
                i += 1
                paragraph.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
        i += 1

    def page_frame(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B6C7D0"))
        canvas.setLineWidth(.5)
        canvas.line(50, PAGE_H-40, PAGE_W-50, PAGE_H-40)
        canvas.setFont("CJK",8)
        canvas.setFillColor(colors.HexColor("#5A6B76"))
        canvas.drawString(50,PAGE_H-30,"投资学团队作业 | CAPM 实证分析")
        canvas.drawRightString(PAGE_W-50,29,str(doc.page))
        canvas.drawString(50,29,"修订稿：成员信息、人工复核与退出事件证据待补")
        canvas.restoreState()

    output = ROOT / "output/pdf/CAPM报告_公式排版版.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer,pagesize=A4,rightMargin=50,leftMargin=50,
                            topMargin=54,bottomMargin=47,title="高贝塔股票是否获得更高收益？",
                            author="团队信息待填写")
    doc.build(story,onFirstPage=page_frame,onLaterPages=page_frame)
    count = len(PdfReader(buffer).pages)
    if not 8 <= count <= 10:
        raise ValueError(f"正文 {count} 页，不满足 8-10 页要求，请调整版式后重新导出")
    staged = ROOT / "tmp/pdfs/CAPM报告_排版检查.pdf"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(buffer.getvalue())
    # 先完成生成和页数检查，再替换正式产物，避免失败时破坏旧 PDF。
    staged.replace(output)
    print(f"PDF: {output} ({count} pages)")


if __name__ == "__main__":
    main()
