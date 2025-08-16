
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def build_report(pdf_path: str, title: str, insights: list[str], figures: list[str]):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, height-2*cm, title)
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, height-2.6*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y = height-3.5*cm
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, y, "Key Insights:")
    y -= 0.8*cm
    c.setFont("Helvetica", 10)
    for line in insights[:12]:
        c.drawString(2.5*cm, y, f"• {line[:110]}")
        y -= 0.6*cm
        if y < 4*cm:
            c.showPage(); y = height-2*cm
    # add figures
    for fig in figures:
        if os.path.exists(fig):
            c.showPage()
            c.drawImage(fig, 2*cm, 4*cm, width-4*cm, height-8*cm, preserveAspectRatio=True, anchor='c')
            c.setFont("Helvetica", 10)
            c.drawString(2*cm, 3.2*cm, os.path.basename(fig))
    c.save()
