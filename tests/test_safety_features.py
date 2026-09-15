import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from backend.app.database import Database
from backend.app.services.safety_service import aggregate

class SafetyFeaturesTests(unittest.TestCase):
    def test_gps_is_scoped_fresh_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Database(Path(tmp)/"test.db");db.initialize();db.initialize()
            d=db.create_or_update_driver("A","Driver",None)
            v=db.create_vehicle("TEST","car")
            t=db.create_trip("T",d["id"],v["id"],None,None)
            sid=db.create_session("video","cpu",1,1,False,t["id"])
            now=datetime(2026,9,10,4,tzinfo=timezone.utc)
            e={"type":"phone","frame_index":1,"timestamp_seconds":1,"confidence":.9,"risk_score":30}
            db.add_vehicle_location(t["id"],10,106,None,None,(now-timedelta(seconds=10)).isoformat(),"gps-simulated")
            with patch("backend.app.database.utc_now",return_value=now.isoformat()): db.add_event(sid,e)
            saved=db.list_events(sid)[0]
            self.assertEqual(saved["latitude"],10);self.assertEqual(saved["gps_source"],"gps-simulated")
            db.add_vehicle_location(t["id"],11,107,None,None,(now+timedelta(seconds=1)).isoformat(),"gps-simulated")
            self.assertEqual(db.list_events(sid)[0]["latitude"],10)
            with patch("backend.app.database.utc_now",return_value=(now+timedelta(minutes=3)).isoformat()): db.add_event(sid,e)
            self.assertIsNone(db.list_events(sid)[1]["latitude"])
            t2=db.create_trip("T2",d["id"],v["id"],None,None)
            sid2=db.create_session("video","cpu",1,1,False,t2["id"])
            db.add_event(sid2,e);self.assertIsNone(db.list_events(sid2)[0]["latitude"])

    def test_score_deduplicates_fatigue_and_clips_night_hours(self):
        sessions=[{"id":1,"driver_id":1,"driver_name":"A","started_at":"2026-09-09T16:30:00+00:00","video_duration_seconds":7200}]
        events=[{"id":i+1,"session_id":1,"driver_id":1,"event_type":kind,"timestamp_seconds":sec,"occurred_at":"2026-09-09T17:05:00+00:00"} for i,(kind,sec) in enumerate([("drowsy",0),("microsleep",0),("high_perclos",1),("phone",2),("drowsy",31)])]
        data=aggregate(sessions,events,"day","2026-09-10")
        self.assertEqual(data["drivers"][0]["safety_score"],72)
        self.assertEqual(data["drivers"][0]["night_monitoring_hours"],1.5)
        self.assertEqual(data["total_events"],5)
        self.assertEqual(aggregate(sessions,events,"day","2026-09-11")["drivers"],[])

    def test_score_never_negative(self):
        events=[{"id":i,"session_id":1,"driver_id":1,"event_type":"phone","timestamp_seconds":i*31,"occurred_at":"2026-09-10T00:00:00Z"} for i in range(20)]
        self.assertEqual(aggregate([],events,"day","2026-09-10")["drivers"][0]["safety_score"],0)

if __name__=="__main__":unittest.main()
