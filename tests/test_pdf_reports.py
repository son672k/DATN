import csv
import io
import tempfile
import unittest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.app.api.routes import create_router
from backend.app.database import Database
from backend.app.security import hash_password
from backend.app.services.pdf_report_service import daily_pdf,session_pdf,save_atomic

class ReportTests(unittest.TestCase):
    def test_pdf_and_admin_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Database(Path(tmp)/"db");db.initialize()
            db.create_or_update_user("admin",hash_password("TestOnly123!"),"Test", "supervisor",None)
            app=FastAPI();app.include_router(create_router(db,type("Manager",(),{"active_session_id":None})()))
            client=TestClient(app)
            token=client.post("/api/auth/token",data={"username":"admin","password":"TestOnly123!"}).json()["access_token"]
            h={"Authorization":f"Bearer {token}"}
            audio=client.get("/api/audio-alerts/distraction.wav",headers=h)
            self.assertEqual(audio.status_code,200);self.assertTrue(audio.content.startswith(b"RIFF"))
            self.assertEqual(client.get("/api/audio-alerts/unknown.wav",headers=h).status_code,404)
            self.assertEqual(client.get("/api/admin/analytics").status_code,401)
            self.assertEqual(client.get("/api/admin/analytics?day=invalid",headers=h).status_code,422)
            self.assertEqual(client.get("/api/admin/reports/daily.pdf?day=invalid",headers=h).status_code,422)
            self.assertEqual(client.get("/api/sessions/999/report.pdf",headers=h).status_code,404)
            self.assertEqual(client.get("/api/sessions/999/report.xlsx",headers=h).status_code,404)
            r=client.get("/api/admin/reports/daily.pdf?day=2026-09-10",headers=h)
            self.assertEqual(r.status_code,200);self.assertTrue(r.content.startswith(b"%PDF"))
            driver=db.create_or_update_driver("PDF-DRV","Tài xế PDF",None)
            vehicle=db.create_vehicle("PDF-001","car")
            trip=db.create_trip("PDF-TRIP",driver["id"],vehicle["id"],None,None)
            sid=db.create_session("test","cpu",1,1,False,trip["id"])
            db.add_frame_metrics(sid,[{"frame_index":4,"timestamp_seconds":0.2,"risk_score":35,"severity":"warning","warnings":"distraction","processing_fps":20.0}])
            db.add_event(sid,{"type":"distraction","frame_index":4,"timestamp_seconds":0.2,"confidence":0.8,"risk_score":35})
            csv_response=client.get(f"/api/sessions/{sid}/report.csv",headers=h)
            self.assertEqual(csv_response.status_code,200)
            csv_text=csv_response.content.decode("utf-8-sig")
            self.assertTrue(csv_text.startswith("sep=;\r\n"))
            rows=list(csv.reader(io.StringIO(csv_text.split("\r\n",1)[1]),delimiter=";"))
            self.assertIn("row_type",rows[0])
            self.assertIn("timestamp_seconds",rows[0])
            self.assertIn("driver_name",rows[0])
            self.assertEqual(len(rows),3)
            self.assertTrue(all(len(row)==len(rows[0]) for row in rows[1:]))
            self.assertEqual({row[0] for row in rows[1:]},{"event","metric"})
            filtered=client.get(f"/api/sessions/{sid}/report.csv?row_type=event&sort_by=risk_score&order=desc",headers=h)
            self.assertEqual(filtered.status_code,200)
            filtered_rows=list(csv.reader(io.StringIO(filtered.content.decode("utf-8-sig").split("\r\n",1)[1]),delimiter=";"))
            self.assertEqual(len(filtered_rows),2)
            self.assertEqual(filtered_rows[1][0],"event")
            self.assertEqual(client.get(f"/api/sessions/{sid}/report.csv?sort_by=invalid",headers=h).status_code,422)
            content=session_pdf(db,sid,Path(tmp));target=Path(tmp)/"reports/report.pdf"
            save_atomic(target,content);self.assertEqual(target.read_bytes(),content)
            self.assertFalse(list(target.parent.glob("*.tmp")))

if __name__=="__main__":unittest.main()
