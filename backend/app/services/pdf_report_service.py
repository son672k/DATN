from __future__ import annotations
import io
import os
import threading
from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from .safety_service import aggregate, stamp, window

_LOCK = threading.Lock()

def build_pdf(title, summary, events, root):
    with _LOCK:
        if "DMS" not in pdfmetrics.getRegisteredFontNames():
            paths = [os.getenv("DMS_PDF_FONT", ""), "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
            font = next((p for p in paths if p and Path(p).is_file()), None)
            if not font:
                raise RuntimeError("Cần font Unicode: đặt DMS_PDF_FONT tới file TTF hỗ trợ tiếng Việt")
            pdfmetrics.registerFont(TTFont("DMS", font))
    out = io.BytesIO()
    style = ParagraphStyle("DMS", fontName="DMS", fontSize=10, leading=15, spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=style, fontSize=17, leading=23, spaceAfter=14)
    p = lambda text: Paragraph(escape(str(text)), style)
    story = [Paragraph(escape(title), heading), p("Hệ thống DMS · GPS mô phỏng · Thời gian trình bày UTC+7")]
    story.extend(p(line) for line in summary)
    story.append(Spacer(1, 12))
    rows = [[p(x) for x in ["Sự kiện", "Thời gian", "Rủi ro", "GPS"]]]
    for e in events:
        when = stamp(e.get("occurred_at"))
        from .safety_service import LOCAL
        gps = "Không có điểm GPS hợp lệ" if e.get("latitude") is None else f"{e['latitude']:.6f}, {e['longitude']:.6f} ({e.get('gps_source') or 'không rõ nguồn'})"
        rows.append([p(f"#{e['id']} {e['event_type']}"), p(when.astimezone(LOCAL).strftime("%d/%m/%Y %H:%M:%S") if when else "—"), p(e.get("risk_score",0)),p(gps)])
    table = Table(rows, colWidths=[120,115,45,235], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8edf3")),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d9d9d9")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6)]))
    story.append(table)
    if not events:
        story.append(p("Không có cảnh báo trong phạm vi báo cáo."))
    root = Path(root).resolve()
    for e in events:
        raw = e.get("snapshot_path")
        if not raw:
            continue
        path = (root / raw).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            story.append(p(f"Ảnh sự kiện #{e['id']} chưa có hoặc không khả dụng."))
            continue
        try:
            picture = Image(str(path))
            ratio = min(480 / picture.imageWidth, 250 / picture.imageHeight)
            picture.drawWidth = picture.imageWidth * ratio
            picture.drawHeight = picture.imageHeight * ratio
            story.extend([Spacer(1,12),KeepTogether([p(f"Ảnh minh chứng sự kiện #{e['id']} · {e['event_type']}"),picture])])
        except Exception:
            story.append(p(f"Không đọc được ảnh sự kiện #{e['id']}."))
    def footer(canvas, doc):
        canvas.setFont("DMS",9)
        canvas.drawRightString(A4[0]-40,22,f"Trang {doc.page}")
    SimpleDocTemplate(out,pagesize=A4,leftMargin=40,rightMargin=40,topMargin=40,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    return out.getvalue()

def session_pdf(database, session_id, root):
    s=database.get_session(session_id)
    if s is None:
        raise ValueError("Không tìm thấy phiên")
    events=database.list_events(session_id)
    return build_pdf(f"Báo cáo phiên giám sát {session_id}",[
        f"Tài xế: {s.get('driver_name') or 'Chưa gán'} · Trạng thái: {s['status']}",
        f"Thời gian giám sát: {float(s.get('video_duration_seconds') or 0)/3600:.3f} giờ · Tổng cảnh báo: {len(events)}",
        f"Rủi ro trung bình: {s.get('avg_risk_score',0)}/100 · FPS trung bình: {s.get('avg_fps',0)}",
        "Điểm an toàn theo ngày/tuần/tháng được trình bày tại mục thống kê. Điểm rủi ro không phải xác suất tai nạn."
    ], events, root)

def daily_pdf(database, day, root):
    sessions,events=database.safety_records()
    data=aggregate(sessions,events,"day",day)
    start,end=window("day",day)
    selected=[e for e in events if (t:=stamp(e.get("occurred_at"))) and start<=t<end]
    lines=[data["policy"],f"Tổng cảnh báo: {len(selected)}"]
    lines += [f"{d['driver_name']}: {d['safety_score']}/100 ({d['rating']}); giám sát {d['monitoring_hours']} giờ; ban đêm {d['night_monitoring_hours']} giờ; {d['total_events']} cảnh báo" for d in data['drivers']]
    return build_pdf(f"Báo cáo an toàn ngày {day}",lines,selected,root)

def save_atomic(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:
        temp.write_bytes(content);temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
