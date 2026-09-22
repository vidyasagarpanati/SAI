"""Narration engine shared by S8 (write) and S9 (repair).

Each section is generated, then checked by the agent itself before it is
accepted: JSON schema, evidence grounding, and the section's own rules. A
rejected output goes back to the model once more with the exact violations
listed. Temperature is 0, so the feedback is what changes the answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import jsonschema

from archery import grounding
from archery.io_guard import guarded_open, guarded_path
from archery.phase_defs import ORDER
from archery.report_spec import (CALLS, PRESCRIPTIVE, SKIP_FIELDS, build_prompt, local_checks,
                                 schema_for, valid_ids)

EVIDENCE_LINE = re.compile(r"^\{\{([\w.\-]+)\}\} = ", re.M)
NUMBER_HINT = ("Remove every typed number that is not a {{KEY}}: no thresholds, targets, "
               "approximations or restated values. Cite the {{KEY}} instead, or drop the number.")


class Narrator:
    def __init__(self, ctx, llm=None):
        self.ctx = ctx
        self.metrics = ctx.read_json("05_metrics.json")
        self.manifest = ctx.read_json("06_manifest.json")
        self.ev = self.metrics["evidence_index"]
        self.dir = guarded_path(ctx.narrative_dir)
        (self.dir / "cache").mkdir(parents=True, exist_ok=True)
        if llm is None:
            from archery.llm import OllamaClient
            llm = OllamaClient(ctx.cfg, self.dir / "cache")
        self.llm = llm
        self.max_retries = int(ctx.cfg.get("llm.max_retries_per_section", 2))
        caps = llm.capabilities()
        self.vision = "vision" in caps and bool(ctx.cfg.get("llm.send_key_frames_to_model", True))
        seen = {p["phase"] for s in self.metrics["phase_timeline"] for p in s["phases"] if p["detected"]}
        self.detected = [p for p in ORDER if p in seen]
        banned_file = ctx.cfg.root / "config" / "banned_phrases.txt"
        self.banned = [ln.strip().lower() for ln in banned_file.read_text(encoding="utf-8").splitlines()
                       if ln.strip() and not ln.startswith("#")] if banned_file.is_file() else []
        self.outputs: dict = {}
        self.log: list[dict] = []
        self.links: list[str] = []
        self._frame_w = None

    # ------------------------------------------------------------ helpers
    def shots_for(self, phase: str) -> list[int]:
        return [s["shot"] for s in self.metrics["phase_timeline"]
                for p in s["phases"] if p["phase"] == phase and p["detected"]]

    def _images(self, phases: list[str]) -> list[bytes]:
        if not self.vision:
            return []
        if self._frame_w is None:
            import json as _j
            fm = _j.loads((self.ctx.run_dir / "01_frames.json").read_text(encoding="utf-8"))
            first = sorted(Path(fm["frames_dir"]).glob("f*.jpg"))[0]
            self._frame_w = cv2.imread(str(first)).shape[1]
        max_w = int(self.ctx.cfg.get("report.model_image_max_width", 768))
        out = []
        for ph in phases:
            f = next((m for m in self.manifest["frames"] if m["phase"] == ph), None)
            if not f:
                continue
            img = cv2.imread(f["file"])[:, :self._frame_w]      # athlete only, no side panel
            if img.shape[1] > max_w:
                img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append(buf.tobytes())
        return out

    def digest_view(self) -> dict:
        """Earlier outputs as later sections see them: s04 without per-shot rows."""
        view = dict(self.outputs)
        if "s04_phase" in view:
            view["s04_phase"] = {ph: {"analysis": o["analysis"], "coaching_implication": o["coaching_implication"]}
                                 for ph, o in view["s04_phase"].items()}
        return view

    def schema(self, sid: str, phase: str | None = None) -> dict:
        return schema_for(sid, {"detected_phases": self.detected, "phase": phase,
                                "ids": valid_ids(self.outputs)})

    def check(self, sid: str, out: dict, schema: dict, phase: str | None) -> list[str]:
        problems = []
        try:
            jsonschema.validate(out, schema)
        except jsonschema.ValidationError as exc:
            where = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            problems.append(f"schema: {where}: {exc.message[:200]}")
            return problems
        problems += grounding.check_object(out, self.ev, PRESCRIPTIVE, SKIP_FIELDS)
        problems += local_checks(sid, out, self.detected, phase, self.outputs,
                                 self.shots_for(phase) if phase else None, self.banned)
        return problems

    # ------------------------------------------------------------ generation
    def generate(self, sid: str, deps: list[str], img_phases: list[str], phase: str | None = None,
                 extra_feedback: list[str] | None = None) -> tuple[dict, list[str], int]:
        feedback = list(extra_feedback or [])
        images = self._images([phase] if phase else img_phases)
        out, problems = {}, ["not generated"]
        for attempt in range(self.max_retries + 1):
            system, user, schema = build_prompt(sid, self.ctx.cfg.prompts, self.metrics, self.digest_view(),
                                                deps, self.detected, phase, bool(images))
            if feedback:
                user += ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED. Fix exactly these problems and "
                         "change nothing else:\n- " + "\n- ".join(feedback[:15]))
            out = self.llm.chat_json(system, user, schema, images)
            # Link restated evidence values back to their keys (unambiguous only).
            subset = {k: self.ev[k] for k in EVIDENCE_LINE.findall(user) if k in self.ev}
            out, linked = grounding.link_object(out, subset, self.ev, SKIP_FIELDS, PRESCRIPTIVE)
            self.links += [f"{sid}{'/' + phase if phase else ''}: {x}" for x in linked]
            problems = self.check(sid, out, schema, phase)
            self.log.append({"section": sid, "phase": phase, "attempt": attempt,
                             "auto_linked": linked[:30], "problems": problems[:20],
                             "images": len(images)})
            if any("typed outside" in p for p in problems):
                problems = problems + [NUMBER_HINT]
            if not problems:
                return out, [], attempt
            feedback = problems
        return out, problems, self.max_retries

    def save(self, name: str, payload) -> Path:
        path = self.dir / f"{name}.json"
        with guarded_open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return path

    def run_all(self) -> dict:
        failed: dict[str, list[str]] = {}
        retries = 0
        for sid, deps, img_phases in CALLS:
            if sid == "s04_phase":
                per = {}
                for ph in self.detected:
                    out, probs, att = self.generate(sid, deps, img_phases, phase=ph)
                    per[ph] = out
                    retries += att
                    if probs:
                        failed[f"s04_phase/{ph}"] = probs
                    self.save(f"s04_phase_{ph}", out)
                self.outputs[sid] = per
                continue
            out, probs, att = self.generate(sid, deps, img_phases)
            retries += att
            self.outputs[sid] = out
            if probs:
                failed[sid] = probs
            self.save(sid, out)
        self.save("all_sections", self.outputs)
        self.save("generation_log", self.log)
        self.save("auto_links", self.links)
        usage = dict(getattr(self.llm, "usage", {}))
        usage["vision_used"] = self.vision
        self.save("usage", usage)
        return {"failed": failed, "retries": retries, "usage": usage, "auto_linked": len(self.links)}

    def load_saved(self) -> None:
        self.outputs = json.loads((self.dir / "all_sections.json").read_text(encoding="utf-8"))
