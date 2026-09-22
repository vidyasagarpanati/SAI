"""LLM clients.

OllamaClient  the real one: localhost only, temperature 0, fixed seed, JSON
              schema enforced by Ollama's structured outputs, reasoning tokens
              off, every response cached on disk by a hash of the full request.
FakeLLM       deterministic stand-in for tests. Writes schema-valid sections
              that cite evidence keys, so S8, S9 and S10 can be exercised
              without a GPU or a model.

Token accounting: every real call records prompt and completion token counts,
which S8 totals into 08_narrative/usage.json.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


class LLMError(RuntimeError):
    pass


class LLMBadOutput(LLMError):
    """The model answered, but not with complete valid JSON (usually truncated).
    Retryable: the narrator sends it back with a request to be more concise."""
    def __init__(self, msg: str, raw: str = "", done_reason: str | None = None):
        super().__init__(msg)
        self.raw, self.done_reason = raw, done_reason


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


class OllamaClient:
    def __init__(self, cfg, cache_dir: Path):
        self.cfg = cfg
        self.base = cfg.get("llm.base_url", "http://localhost:11434").rstrip("/")
        self.model = cfg.get("llm.model")
        if not self.model:
            raise LLMError("config/pipeline.yaml llm.model is not set. Run `ollama list` and set it.")
        self.cache_dir = cache_dir
        self.timeout = float(cfg.get("llm.request_timeout_s", 300))
        self._caps: set[str] | None = None
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        self.last: dict = {}

    def capabilities(self) -> set[str]:
        if self._caps is None:
            import httpx
            try:
                r = httpx.post(f"{self.base}/api/show", json={"model": self.model}, timeout=30)
                r.raise_for_status()
                self._caps = set(r.json().get("capabilities") or [])
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"Ollama at {self.base} not reachable or model {self.model!r} "
                               f"unknown: {exc}") from exc
        return self._caps

    def chat_json(self, system: str, user: str, schema: dict,
                  images: list[bytes] | None = None) -> dict:
        import httpx
        opts = {"temperature": float(self.cfg.get("llm.temperature", 0.0)),
                "seed": int(self.cfg.get("llm.seed", 0)),
                "top_p": float(self.cfg.get("llm.top_p", 1.0)),
                "num_ctx": int(self.cfg.get("llm.num_ctx", 16384)),
                "num_predict": int(self.cfg.get("llm.num_predict", 4096))}
        think = bool(self.cfg.get("llm.think", False))
        key = hashlib.sha256(json.dumps({
            "model": self.model, "opts": opts, "think": think, "system": system, "user": user,
            "schema": schema, "images": [hashlib.sha256(b).hexdigest() for b in images or []],
        }, sort_keys=True).encode()).hexdigest()
        cache = self.cache_dir / f"{key[:32]}.json"
        if cache.is_file():
            self.usage["cached"] += 1
            blob = json.loads(cache.read_text(encoding="utf-8"))
            self.last = {"seconds": 0.0, "prompt_tokens": blob.get("prompt_tokens") or 0,
                         "completion_tokens": blob.get("completion_tokens") or 0, "cached": True}
            return blob["parsed"]

        msg = {"role": "user", "content": user}
        if images:
            msg["images"] = [base64.b64encode(b).decode() for b in images]
        payload = {"model": self.model, "stream": False, "format": schema, "options": opts,
                   "messages": [{"role": "system", "content": system}, msg]}
        if not think:
            payload["think"] = False
        t0 = time.time()
        r = httpx.post(f"{self.base}/api/chat", json=payload, timeout=self.timeout)
        if r.status_code == 400 and "think" in r.text.lower():
            payload.pop("think", None)       # model or server without thinking support
            r = httpx.post(f"{self.base}/api/chat", json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise LLMError(f"Ollama returned {r.status_code}: {r.text[:500]}")
        body = r.json()
        content = body.get("message", {}).get("content", "")
        reason = body.get("done_reason")
        try:
            parsed = _parse_json(content)
        except Exception as exc:  # noqa: BLE001
            self.usage["calls"] += 1
            why = ("cut off at the output limit (num_predict)" if reason == "length"
                   else f"not valid JSON (done_reason={reason})")
            raise LLMBadOutput(f"Model reply {why}; prompt {body.get('prompt_eval_count')} tokens, "
                               f"reply {body.get('eval_count')} tokens. Start: {content[:160]!r}",
                               raw=content, done_reason=reason) from exc
        dt = time.time() - t0
        self.last = {"seconds": round(dt, 1), "prompt_tokens": int(body.get("prompt_eval_count") or 0),
                     "completion_tokens": int(body.get("eval_count") or 0), "cached": False}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += int(body.get("prompt_eval_count") or 0)
        self.usage["completion_tokens"] += int(body.get("eval_count") or 0)
        self.usage["seconds"] += dt
        from archery.io_guard import guarded_open
        with guarded_open(cache, "w", encoding="utf-8") as fh:
            json.dump({"model": self.model, "options": opts, "seconds": round(dt, 1),
                       "prompt_tokens": body.get("prompt_eval_count"),
                       "completion_tokens": body.get("eval_count"),
                       "raw": content, "parsed": parsed}, fh, indent=1)
        return parsed


# ---------------------------------------------------------------- fake for tests
class FakeLLM:
    """Deterministic, schema-valid sections built from the EVIDENCE keys in the
    prompt. ``fabricate`` injects a typed number once, to prove S8 catches it."""

    def __init__(self, fabricate_in: str | None = None, vision: bool = True,
                 truncate_in: str | None = None):
        self.fabricate_in = fabricate_in
        self.truncate_in = truncate_in
        self._truncated = False
        self._fabricated = False
        self.vision = vision
        self.calls: list[str] = []
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def capabilities(self) -> set[str]:
        return {"completion", "vision"} if self.vision else {"completion"}

    def chat_json(self, system, user, schema, images=None) -> dict:
        sid = re.search(r"SECTION: (\w+)", user).group(1)
        phase = (re.search(r"PHASE: (\w+)", user) or [None, None])[1]
        keys = re.findall(r"^\{\{([\w.\-]+)\}\} = ", user, flags=re.M)
        ctxj = json.loads(re.search(r"CONTEXT\n(\{.*\})", user).group(1))
        prior = re.findall(r'"id":"([SWE]\d+)"', user)
        self.calls.append(sid + (f"/{phase}" if phase else ""))
        self.usage["calls"] += 1
        self.last = {"seconds": 0.0, "prompt_tokens": len(user) // 4,
                     "completion_tokens": 200, "cached": False}
        self.last_user = user
        self.last_schema = schema
        if self.truncate_in and sid == self.truncate_in and not self._truncated:
            self._truncated = True
            raise LLMBadOutput("cut off at the output limit (num_predict)",
                               raw='{"phase": "STANCE", "analysis": [{"point": "Stance wid', done_reason="length")
        k = lambda i=0: keys[i % len(keys)] if keys else None  # noqa: E731
        cite = lambda i=0: (f"value {{{{{k(i)}}}}}" if k(i) else "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO")  # noqa: E731
        ek = lambda i=0: [k(i)] if k(i) else []  # noqa: E731
        detected = ctxj["detected_phases"]
        W = [p for p in prior if p.startswith("W")] or ["W1"]
        S = [p for p in prior if p.startswith("S")] or ["S1"]
        out = self._build(sid, phase, ctxj, detected, cite, ek, S, W)
        if self.fabricate_in == sid and not self._fabricated:
            self._fabricated = True
            _inject_number(out)
        return out

    def _build(self, sid, phase, ctxj, detected, cite, ek, S, W):
        pt = lambda i, lvl="MEASURED": {"point": f"Observation: {cite(i)}.", "evidence_level": lvl,  # noqa: E731
                                        "confidence": "HIGH", "evidence_keys": ek(i)}
        if sid == "s02_quality":
            return {f: f"{f.replace('_', ' ')} adequate for landmark tracking" for f in
                    ("camera_angle", "lighting", "athlete_visibility", "occlusion", "motion_blur",
                     "clothing_interference")} | {"limitations": ["Single camera view limits depth."]}
        if sid == "s04_phase":
            shots = ctxj["phase"]["shots_with_phase"]
            return {"phase": phase, "criteria_groups": ["E"], "analysis": [pt(0), pt(1)],
                    "frame_rows": [{"shot": s, "observation": f"Observed at {cite(2)}.",
                                    "interpretation": "Consistent with a stable hold.",
                                    "evidence_level": "MEASURED", "confidence": "HIGH"} for s in shots],
                    "coaching_implication": "Maintain the current alignment through this phase."}
        if sid == "s05_biomech":
            return {"findings": [{"topic": t, "finding": cite(i), "evidence_level": "MEASURED",
                                  "confidence": "MEDIUM", "evidence_keys": ek(i)}
                                 for i, t in enumerate(["Trunk inclination", "Elbow alignment",
                                                        "Shoulder alignment", "Body sway"])]}
        if sid == "s06_consistency":
            return {"effects": [{"aspect": a, "explanation": "Limited by the number of shots.",
                                 "confidence": "LOW"} for a in
                                ["Accuracy", "Grouping", "Arrow flight", "Repeatability", "Pressure performance"]],
                    "notes": "Shot-to-shot consistency requires at least three shots."}
        if sid == "s08_equipment":
            aspects = ["Bow fit", "Draw length", "Arrow length", "Arrow spine indications",
                       "Stabilizer behavior", "String alignment", "Nocking-point clues",
                       "Grip configuration", "Bow torque", "Equipment-induced movement"]
            return {"items": [{"aspect": a, "classification": "NOT ASSESSABLE",
                               "observation": "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO",
                               "confidence": "LOW"} for a in aspects]}
        if sid == "s09_errors":
            return {"errors": [{"id": "E1", "rank": 1, "error": "Draw elbow position", "phase": detected[0],
                                "timestamp_key": "shot1.AIM.start_t_s", "evidence": cite(0),
                                "cause": "Scapular setup", "biomechanical_effect": "Alters line of force",
                                "performance_effect": "Vertical spread", "severity": "MODERATE",
                                "correction": "Rehearse alignment at anchor", "confidence": "MEDIUM",
                                "evidence_keys": ek(0)}]}
        if sid == "s10_injury":
            return {"risks": [{"factor": "Shoulder elevation", "evidence": cite(0),
                               "possible_concern": "Potential movement-related risk factor for shoulder load",
                               "severity": "MINOR", "corrective_strategy": "Scapular depression drills",
                               "confidence": "LOW", "evidence_keys": ek(0)}]}
        if sid == "s11_framework":
            from archery.report_spec import CATEGORIES
            return {"categories": [{"category": c, "score": None if c in "DG" else 7,
                                    "evidence": cite(i) if c not in "DG" else "NOT RELIABLY ASSESSABLE FROM AVAILABLE VIDEO",
                                    "main_limitation": "See Section 9", "recommendation": "Maintain",
                                    "confidence": "MEDIUM", "evidence_keys": [] if c in "DG" else ek(i)}
                                   for i, (c, _) in enumerate(CATEGORIES)]}
        if sid == "s12_scorecard":
            sc = {"score": 7, "justification": cite(0), "confidence": "MEDIUM", "evidence_keys": ek(0)}
            return {"phases": [{"phase": p, "score": 7, "justification": cite(i), "confidence": "MEDIUM",
                                "evidence_keys": ek(i)} for i, p in enumerate(detected)],
                    "shot_consistency": dict(sc), "biomechanical_efficiency": dict(sc),
                    "movement_stability": dict(sc), "overall_technique": dict(sc)}
        if sid == "s13_strengths":
            return {"items": [{"id": "S1", "strength": "Stable bow arm", "evidence": cite(0),
                               "biomechanical_reason": "Extended elbow resists collapse",
                               "performance_benefit": "Consistent line", "maintenance_strategy": "Keep drilling",
                               "confidence": "HIGH", "evidence_keys": ek(0)}]}
        if sid == "s14_weaknesses":
            return {"items": [{"id": "W1", "rank": 1, "weakness": "Draw elbow alignment", "evidence": cite(0),
                               "likely_cause": "Scapular setup", "performance_consequence": "Vertical spread",
                               "correction": "Alignment drill", "priority": "HIGH", "confidence": "MEDIUM",
                               "evidence_keys": ek(0), "related_errors": ["E1"]}]}
        if sid == "s15_priorities":
            return {"priorities": [{"rank": r, "weakness_ref": W[0], "issue": "Draw elbow alignment" if r == 1 else "Supporting correction",
                                    "current_problem": cite(0), "evidence": cite(0),
                                    "timestamp_key": "shot1.AIM.start_t_s", "shot_phase": detected[0],
                                    "why_it_matters": "Line of force", "biomechanical_consequence": "Torque",
                                    "performance_consequence": "Spread", "corrective_strategy": "Blank bale",
                                    "technical_drill": "Mirror anchor drill", "sets": "3 sets",
                                    "repetitions": "10 reps", "frequency": "4x per week",
                                    "progression": "Add hold time after 2 weeks",
                                    "success_metric": f"Track {cite(0)}", "confidence": "MEDIUM"}
                                   for r in range(1, 6)]}
        if sid == "s16_training_coaching":
            rec = {"purpose": "Stabilise anchor", "exercise": "Blank bale shots", "sets": "3 sets",
                   "repetitions": "12 reps", "frequency": "3x per week", "coaching_cue": "Hold and expand",
                   "progression": "Increase to 4 sets", "addresses": [W[0]]}
            cp = {"recommendation": "Rehearse anchor", "addresses": [W[0]]}
            return {"technical": [rec], "strength_conditioning": [rec], "mental": [], "warm_up": [rec],
                    "immediate": [cp], "short_term": [cp], "long_term": [cp]}
        if sid == "s18_projection_final":
            return {"projection": [{"aspect": a, "projection": "Likely to improve if W1 is corrected.",
                                    "confidence": "LOW"} for a in
                                   ["Consistency", "Grouping", "Stability", "Repeatability", "Movement efficiency"]],
                    "does_well": [{"text": "Stable bow arm", "ref": S[0]}] * 3,
                    "fix_first": [{"text": "Draw elbow alignment", "ref": W[0]}] * 3,
                    "single_correction": {"text": "Align the draw elbow behind the arrow line", "ref": W[0]},
                    "key_cue": "Elbow behind the arrow", "final_assessment": "Sound base with one clear fix."}
        if sid == "s01_executive":
            return {"technical_level": "Developing (see Section 12)",
                    "strongest_characteristic": {"text": "Stable bow arm", "ref": S[0]},
                    "most_important_weakness": {"text": "Draw elbow alignment", "ref": W[0]},
                    "main_consistency_limitation": "Single shot limits consistency assessment",
                    "main_biomechanical_limitation": "Draw elbow alignment",
                    "main_injury_risk_factor": "Potential movement-related risk factor at the shoulder",
                    "most_important_correction": {"text": "Align the draw elbow", "ref": W[0]},
                    "expected_improvement_opportunity": "Tighter vertical grouping"}
        raise KeyError(sid)


def _inject_number(out: Any) -> None:
    """Replace the first prose string with one containing a typed measurement."""
    def visit(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and len(v) > 12 and k not in ("id", "ref", "phase"):
                    o[k] = v + " The elbow sat at 172.4 degrees."
                    return True
                if visit(v):
                    return True
        if isinstance(o, list):
            return any(visit(x) for x in o)
        return False
    visit(out)
