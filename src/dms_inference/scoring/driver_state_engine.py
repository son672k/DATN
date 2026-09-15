from __future__ import annotations


class DriverStateEngine:
    def __init__(
        self,
        drowsy_score: int = 50,
        high_perclos_score: int = 25,
        phone_score: int = 30,
        cigarette_score: int = 15,
        no_seatbelt_score: int = 20,
        yawn_score: int = 15,
        microsleep_score: int = 50,
        distraction_score: int = 20,
        perclos_threshold: float = 0.40,
        infer_no_seatbelt_from_absence: bool = False,
    ):
        self.scores = {
            "drowsy": int(drowsy_score),
            "high_perclos": int(high_perclos_score),
            "phone": int(phone_score),
            "cigarette": int(cigarette_score),
            "no_seatbelt": int(no_seatbelt_score),
            "yawn": int(yawn_score),
            "microsleep": int(microsleep_score),
            "distraction": int(distraction_score),
        }
        self.perclos_threshold = float(perclos_threshold)
        self.infer_no_seatbelt_from_absence = bool(infer_no_seatbelt_from_absence)

    def evaluate(self, temporal: dict, behavior: dict, perclos: dict, geometry: dict | None = None) -> dict:
        warnings = []
        drowsy = temporal.get("ready") and temporal.get("state") == "drowsy"
        high_perclos = perclos.get("ready") and perclos.get("value", 0.0) >= self.perclos_threshold
        microsleep = bool(geometry and geometry.get("microsleep"))
        if microsleep:
            warnings.append("microsleep")
        elif high_perclos:
            warnings.append("high_perclos")
        elif drowsy:
            warnings.append("drowsy")
        phone = bool(behavior.get("phone"))
        cigarette = bool(behavior.get("cigarette"))
        if phone:
            warnings.append("phone")
        if cigarette:
            warnings.append("cigarette")
        if geometry and geometry.get("yawning"):
            warnings.append("yawn")
        attention_observable = bool(
            geometry
            and not geometry.get("eye_closed")
            and not behavior.get("eye_closed")
            and not geometry.get("yawning")
        )
        if attention_observable and not (phone or cigarette) and (geometry.get("head_distracted") or geometry.get("gaze_distracted")):
            warnings.append("distraction")
        if self.infer_no_seatbelt_from_absence and not behavior.get("seatbelt", False):
            warnings.append("no_seatbelt")
        risk_score = min(100, sum(self.scores[name] for name in warnings))
        severity = "danger" if risk_score >= 60 else "warning" if risk_score >= 25 else "normal"
        return {"risk_score": risk_score, "severity": severity, "warnings": warnings}
