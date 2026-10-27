import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from garminconnect.workout import (
    ConditionType,
    CyclingWorkout,
    ExecutableStep,
    RepeatGroup,
    RunningWorkout,
    SportType,
    StepType,
    SwimmingWorkout,
    TargetType,
    WorkoutSegment,
)

ZONE_PCT_MAP = {
    1: (0.45, 0.55),
    2: (0.56, 0.75),
    3: (0.76, 0.90),
    4: (0.91, 1.05),
    5: (1.06, 1.20),
    6: (1.21, 1.50),
}


def extract_workout_from_ai_response(ai_text: str) -> tuple[str | None, str | None, str]:
    """Haal een workout-blok, titel en sport ('Ride'/'Run'/'Swim') uit een AI Coach bericht."""
    if not ai_text:
        return None, None, "Ride"

    sport = _detect_sport_from_text(ai_text)
    code_blocks = re.findall(
        r"```(?:text|markdown|intervals|workout)?\s*\n(.*?)```",
        ai_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for block in code_blocks:
        if re.search(r"^\s*-\s*\d+\s*(?:m|min|s|sec|h|u|km|meter)\b", block, flags=re.MULTILINE | re.IGNORECASE):
            naam = _extract_title_from_text(ai_text, block)
            block_sport = _detect_sport_from_text(block) or sport
            clean_block = _strip_metadata_lines(block)
            return naam, clean_block, block_sport

    lines = ai_text.splitlines()
    workout_lines: list[str] = []
    in_workout = False
    for line in lines:
        stripped = line.strip()
        is_step = bool(re.match(r"^-\s*\d+\s*(?:m|min|s|sec|h|u|\d*:)\b", stripped, flags=re.IGNORECASE))
        is_header = bool(
            re.match(
                r"^(?:warmup|warm-up|warming-up|inrijden|inlopen|main\s*set.*|set\s*\d+\s*x.*|\d+\s*x\b.*|herhaal\s*\d+\s*x.*|cooldown|cool-down|cooling-down|uitrijden|uitlopen|kern.*)$",
                stripped,
                flags=re.IGNORECASE,
            )
        )
        if is_step or is_header:
            in_workout = True
            workout_lines.append(stripped)
        elif in_workout and stripped == "":
            workout_lines.append("")
        elif in_workout and not is_step and not is_header:
            step_count = sum(1 for wl in workout_lines if re.match(r"^-\s*\d+", wl))
            if step_count >= 2:
                break
            else:
                workout_lines = []
                in_workout = False

    step_count = sum(1 for wl in workout_lines if re.match(r"^-\s*\d+", wl))
    if step_count >= 1:
        extracted = "\n".join(workout_lines).strip()
        naam = _extract_title_from_text(ai_text, extracted)
        return naam, extracted, sport

    return None, None, sport


def _strip_metadata_lines(block: str) -> str:
    out = []
    for line in block.splitlines():
        if re.match(r"^\s*(?:naam|titel|workout|sport|type)\s*:", line, flags=re.IGNORECASE):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _detect_sport_from_text(text: str) -> str:
    m = re.search(r"(?:sport|type)\s*:\s*(ride|run|swim|fietsen|wielrennen|hardlopen|lopen|zwemmen)", text, flags=re.IGNORECASE)
    if m:
        val = m.group(1).lower()
        if val in ("run", "hardlopen", "lopen"):
            return "Run"
        if val in ("swim", "zwemmen"):
            return "Swim"
        return "Ride"
    if re.search(r"\b\d+:\d{2}\s*/\s*km\b", text, flags=re.IGNORECASE) or re.search(r"\b(?:hardloop|duurloop|inlopen|uitlopen)\b", text, flags=re.IGNORECASE):
        return "Run"
    return "Ride"


def _extract_title_from_text(full_text: str, block_text: str) -> str:
    for src in (block_text, full_text):
        for line in src.splitlines():
            m = re.match(r"^\s*(?:naam|titel|workout|training|sessie)\s*:\s*(.+)$", line.strip().strip("#*"), flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip().strip("*#\"'")
                if 3 <= len(candidate) <= 70:
                    return candidate

    for line in full_text.splitlines():
        clean = line.strip().strip("#*").strip()
        if any(
            k in clean.lower()
            for k in [
                "sweetspot", "vo2max", "herstel", "zone 2", "duurrit",
                "interval", "threshold", "drempel", "duurloop", "fartlek", "tempo",
            ]
        ):
            if 4 <= len(clean) <= 65 and not clean.startswith("-"):
                return re.sub(r"^(?:workout|training|sessie)\s*:\s*", "", clean, flags=re.IGNORECASE).strip()
    return "AI Garmin Workout"


def _parse_duration_seconds(line: str) -> tuple[int, str]:
    s = re.sub(r"^-\s*", "", line.strip())
    m = re.match(r"^(\d+)\s*(?:h|u|uur)\s*(\d+)\s*(?:m|min|minuten)?\b(.*)$", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60, m.group(3).strip()
    m = re.match(r"^(\d+):(\d{2})(?!\s*/\s*km)\b(.*)$", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)), m.group(3).strip()
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*(?:h|u|uur)\b(.*)$", s, flags=re.IGNORECASE)
    if m:
        return int(round(float(m.group(1).replace(",", ".")) * 3600)), m.group(2).strip()
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*(?:m|min|minuten|mins)\b(.*)$", s, flags=re.IGNORECASE)
    if m:
        return int(round(float(m.group(1).replace(",", ".")) * 60)), m.group(2).strip()
    m = re.match(r"^(\d+)\s*(?:s|sec|seconden|secs)\b(.*)$", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return 0, s


def _parse_targets(rest: str, ftp: int) -> dict[str, Any]:
    watt_low = watt_high = hr_low = hr_high = rpm_low = rpm_high = None
    pct_low = pct_high = pace_mps_low = pace_mps_high = pace_label = None

    m_rpm_range = re.search(r"(\d{2,3})\s*-\s*(\d{2,3})\s*(?:rpm|spm)\b", rest, flags=re.IGNORECASE)
    if m_rpm_range:
        rpm_low, rpm_high = int(m_rpm_range.group(1)), int(m_rpm_range.group(2))
    else:
        m_rpm = re.search(r"(\d{2,3})\s*(?:rpm|spm)\b", rest, flags=re.IGNORECASE)
        if m_rpm:
            r = int(m_rpm.group(1))
            rpm_low, rpm_high = max(40, r - 5), r + 5

    rest_clean = re.sub(r"\d{2,3}(?:\s*-\s*\d{2,3})?\s*(?:rpm|spm)\b", "", rest, flags=re.IGNORECASE)

    m_pace_range = re.search(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*(?:/\s*km|min\s*/\s*km)", rest_clean, flags=re.IGNORECASE)
    if m_pace_range:
        s1 = int(m_pace_range.group(1)) * 60 + int(m_pace_range.group(2))
        s2 = int(m_pace_range.group(3)) * 60 + int(m_pace_range.group(4))
        slow_s, fast_s = max(s1, s2), min(s1, s2)
        pace_mps_low, pace_mps_high = round(1000.0 / slow_s, 3), round(1000.0 / fast_s, 3)
        pace_label = f"{fast_s // 60}:{fast_s % 60:02d} – {slow_s // 60}:{slow_s % 60:02d} /km"
        rest_clean = re.sub(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*(?:/\s*km|min\s*/\s*km)", "", rest_clean, flags=re.IGNORECASE)
    else:
        m_pace = re.search(r"(\d{1,2}):(\d{2})\s*(?:/\s*km|min\s*/\s*km)", rest_clean, flags=re.IGNORECASE)
        if m_pace:
            sec_per_km = int(m_pace.group(1)) * 60 + int(m_pace.group(2))
            slow_s, fast_s = sec_per_km + 8, max(120, sec_per_km - 8)
            pace_mps_low, pace_mps_high = round(1000.0 / slow_s, 3), round(1000.0 / fast_s, 3)
            pace_label = f"{fast_s // 60}:{fast_s % 60:02d} – {slow_s // 60}:{slow_s % 60:02d} /km"
            rest_clean = re.sub(r"(\d{1,2}):(\d{2})\s*(?:/\s*km|min\s*/\s*km)", "", rest_clean, flags=re.IGNORECASE)

    m_hr_range = re.search(r"(\d{2,3})\s*-\s*(\d{2,3})\s*bpm\b", rest_clean, flags=re.IGNORECASE)
    if m_hr_range:
        hr_low, hr_high = int(m_hr_range.group(1)), int(m_hr_range.group(2))
        rest_clean = re.sub(r"(\d{2,3})\s*-\s*(\d{2,3})\s*bpm\b", "", rest_clean, flags=re.IGNORECASE)
    else:
        m_hr = re.search(r"(\d{2,3})\s*bpm\b", rest_clean, flags=re.IGNORECASE)
        if m_hr:
            h = int(m_hr.group(1))
            hr_low, hr_high = max(80, h - 4), h + 4
            rest_clean = re.sub(r"(\d{2,3})\s*bpm\b", "", rest_clean, flags=re.IGNORECASE)

    m_pct_range = re.search(r"(\d{2,3})\s*-\s*(\d{2,3})\s*%", rest_clean)
    if m_pct_range:
        pct_low, pct_high = int(m_pct_range.group(1)) / 100.0, int(m_pct_range.group(2)) / 100.0
        watt_low, watt_high = int(round(ftp * pct_low)), int(round(ftp * pct_high))
    else:
        m_pct = re.search(r"(\d{2,3})\s*%", rest_clean)
        if m_pct:
            p = int(m_pct.group(1)) / 100.0
            pct_low, pct_high = max(0.30, p - 0.03), p + 0.03
            watt_low, watt_high = int(round(ftp * pct_low)), int(round(ftp * pct_high))

    if watt_low is None:
        m_w_range = re.search(r"(\d{2,4})\s*-\s*(\d{2,4})\s*w(?:att)?\b", rest_clean, flags=re.IGNORECASE)
        if m_w_range:
            watt_low, watt_high = int(m_w_range.group(1)), int(m_w_range.group(2))
            pct_low = watt_low / ftp if ftp else None
            pct_high = watt_high / ftp if ftp else None
        else:
            m_w = re.search(r"(\d{2,4})\s*w(?:att)?\b", rest_clean, flags=re.IGNORECASE)
            if m_w:
                w = int(m_w.group(1))
                margin = max(5, int(round(w * 0.03)))
                watt_low, watt_high = max(30, w - margin), w + margin
                pct_low = watt_low / ftp if ftp else None
                pct_high = watt_high / ftp if ftp else None

    if watt_low is None and hr_low is None and pace_mps_low is None:
        m_z = re.search(r"\b(?:z|zone\s*)([1-6])\b", rest_clean, flags=re.IGNORECASE)
        if m_z:
            p1, p2 = ZONE_PCT_MAP.get(int(m_z.group(1)), (0.56, 0.75))
            pct_low, pct_high = p1, p2
            watt_low, watt_high = int(round(ftp * p1)), int(round(ftp * p2))

    avg_watt = int(round((watt_low + watt_high) / 2)) if (watt_low is not None and watt_high is not None) else None
    avg_pct = int(round(((pct_low + pct_high) / 2) * 100)) if (pct_low is not None and pct_high is not None) else None

    return {
        "watt_low": watt_low, "watt_high": watt_high, "avg_watt": avg_watt, "avg_pct": avg_pct,
        "hr_low": hr_low, "hr_high": hr_high, "pace_mps_low": pace_mps_low, "pace_mps_high": pace_mps_high,
        "pace_label": pace_label, "rpm_low": rpm_low, "rpm_high": rpm_high,
    }


def parse_workout_structure(workout_text: str, ftp: int) -> dict[str, Any]:
    lines = workout_text.splitlines()
    blocks: list[dict[str, Any]] = []
    current_section = "auto"
    current_repeat: dict[str, Any] | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_repeat and current_repeat["steps"]:
                blocks.append(current_repeat)
                current_repeat = None
            continue
        if re.match(r"^(?:naam|titel|workout|sport|type)\s*:", line, flags=re.IGNORECASE):
            continue

        m_rep = re.match(r"^(?:main\s*set|kern|set|herhaal|repeat|blok|intervall?en)?\s*(\d+)\s*x\b.*$", line, flags=re.IGNORECASE)
        if not m_rep:
            m_rep = re.search(r"\b(\d+)\s*x\s*:?$", line, flags=re.IGNORECASE) if not line.startswith("-") else None
        if m_rep and not line.startswith("-"):
            if current_repeat and current_repeat["steps"]:
                blocks.append(current_repeat)
            current_repeat = {"type": "repeat", "iterations": max(1, int(m_rep.group(1))), "steps": []}
            current_section = "main"
            continue

        lower = line.lower().rstrip(":")
        if lower in ("warmup", "warm-up", "warming-up", "inrijden", "inlopen"):
            if current_repeat and current_repeat["steps"]:
                blocks.append(current_repeat)
                current_repeat = None
            current_section = "warmup"
            continue
        if lower in ("cooldown", "cool-down", "cooling-down", "uitrijden", "uitlopen"):
            if current_repeat and current_repeat["steps"]:
                blocks.append(current_repeat)
                current_repeat = None
            current_section = "cooldown"
            continue
        if lower in ("main set", "kern", "hoofdblok"):
            if current_repeat and current_repeat["steps"]:
                blocks.append(current_repeat)
                current_repeat = None
            current_section = "main"
            continue

        if line.startswith("-") or re.match(r"^\d+\s*(?:m|min|s|sec|h|u)\b", line, flags=re.IGNORECASE):
            dur_sec, rest = _parse_duration_seconds(line)
            if dur_sec <= 0:
                continue
            step_info = {"type": "step", "section": current_section, "duration_sec": dur_sec, "raw": line, **_parse_targets(rest, ftp)}
            if current_repeat is not None:
                current_repeat["steps"].append(step_info)
            else:
                blocks.append(step_info)

    if current_repeat and current_repeat["steps"]:
        blocks.append(current_repeat)

    total_blocks = len(blocks)
    for idx, blk in enumerate(blocks):
        if blk["type"] == "step":
            sec = blk.get("section", "auto")
            if sec == "warmup" or (sec == "auto" and idx == 0 and total_blocks >= 2):
                blk["step_kind"] = "warmup"
            elif sec == "cooldown" or (sec == "auto" and idx == total_blocks - 1 and total_blocks >= 2):
                blk["step_kind"] = "cooldown"
            else:
                pct = blk.get("avg_pct") or 70
                blk["step_kind"] = "recovery" if (pct < 60 and 0 < idx < total_blocks - 1) else "interval"
        elif blk["type"] == "repeat":
            substeps = blk["steps"]
            for s_idx, s in enumerate(substeps):
                pct = s.get("avg_pct") or 75
                raw_lower = s.get("raw", "").lower()
                if len(substeps) >= 2 and s_idx == len(substeps) - 1 and (pct <= 68 or any(w in raw_lower for w in ["rust", "herstel", "recovery", "dribbel", "easy"])):
                    s["step_kind"] = "recovery"
                else:
                    s["step_kind"] = "interval"

    preview_rows: list[dict[str, Any]] = []
    total_sec = 0
    total_joules = 0.0
    weighted_tss_sum = 0.0
    kind_labels = {"warmup": "🟢 Warm-up", "interval": "🔥 Interval / Kern", "recovery": "💧 Herstel / Rust", "cooldown": "🔵 Cool-down"}

    def _add_step_stats(s: dict[str, Any], mult: int, prefix: str = "") -> None:
        nonlocal total_sec, total_joules, weighted_tss_sum
        d_sec = s["duration_sec"]
        total_sec += d_sec * mult
        avg_w = s.get("avg_watt") or int(round(ftp * 0.65))
        total_joules += avg_w * d_sec * mult
        if_val = (avg_w / ftp) if ftp > 0 else 0.65
        weighted_tss_sum += (((d_sec * mult) * avg_w * if_val) / (ftp * 3600.0) * 100.0) if ftp > 0 else 0.0
        min_str = f"{d_sec // 60}m" if d_sec % 60 == 0 else (f"{d_sec // 60}m {d_sec % 60}s" if d_sec >= 60 else f"{d_sec}s")
        if s.get("pace_label"):
            doel_str = f"🏃 {s['pace_label']}" + (f" ({s['hr_low']}–{s['hr_high']} bpm)" if s.get("hr_low") else "")
        elif s.get("watt_low") and s.get("watt_high"):
            doel_str = f"⚡ {s['watt_low']} – {s['watt_high']} W ({s.get('avg_pct', '')}% FTP)"
        elif s.get("hr_low") and s.get("hr_high"):
            doel_str = f"❤️ {s['hr_low']} – {s['hr_high']} bpm"
        else:
            doel_str = "Vrij / Op gevoel"
        preview_rows.append({
            "Blok": f"{prefix}{kind_labels.get(s.get('step_kind', 'interval'), 'Interval')}",
            "Herhaling": f"{mult}x" if mult > 1 else "1x",
            "Duur per stap": min_str,
            "Doel (Watt / Tempo / HR)": doel_str,
            "Cadans": f"{s['rpm_low']}–{s['rpm_high']} rpm" if s.get("rpm_low") else "—",
        })

    for blk in blocks:
        if blk["type"] == "step":
            _add_step_stats(blk, 1)
        elif blk["type"] == "repeat":
            for s in blk["steps"]:
                _add_step_stats(s, blk["iterations"], prefix="↳ ")

    return {
        "blocks": blocks, "preview_rows": preview_rows, "total_seconds": total_sec,
        "total_minutes": round(total_sec / 60.0, 1), "estimated_kj": int(round(total_joules / 1000.0)),
        "estimated_tss": int(round(weighted_tss_sum)),
    }


def _make_executable_step(step_data: dict[str, Any], step_order: int) -> ExecutableStep:
    kind = step_data.get("step_kind", "interval")
    step_type_map = {
        "warmup": (StepType.WARMUP, "warmup", 1),
        "cooldown": (StepType.COOLDOWN, "cooldown", 2),
        "interval": (StepType.INTERVAL, "interval", 3),
        "recovery": (StepType.RECOVERY, "recovery", 4),
    }
    st_id, st_key, st_order = step_type_map.get(kind, (StepType.INTERVAL, "interval", 3))
    w_low, w_high = step_data.get("watt_low"), step_data.get("watt_high")
    pace_low, pace_high = step_data.get("pace_mps_low"), step_data.get("pace_mps_high")
    hr_low, hr_high = step_data.get("hr_low"), step_data.get("hr_high")

    if w_low is not None and w_high is not None:
        target_type = {"workoutTargetTypeId": TargetType.POWER_ZONE, "workoutTargetTypeKey": "power.zone", "displayOrder": 2}
        val_one, val_two = float(w_low), float(w_high)
    elif pace_low is not None and pace_high is not None:
        target_type = {"workoutTargetTypeId": TargetType.PACE_ZONE, "workoutTargetTypeKey": "pace.zone", "displayOrder": 6}
        val_one, val_two = float(pace_low), float(pace_high)
    elif hr_low is not None and hr_high is not None:
        target_type = {"workoutTargetTypeId": TargetType.HEART_RATE_ZONE, "workoutTargetTypeKey": "heart.rate.zone", "displayOrder": 4}
        val_one, val_two = float(hr_low), float(hr_high)
    else:
        target_type = {"workoutTargetTypeId": TargetType.NO_TARGET, "workoutTargetTypeKey": "no.target", "displayOrder": 1}
        val_one = val_two = None

    rpm_low, rpm_high = step_data.get("rpm_low"), step_data.get("rpm_high")
    if rpm_low is not None and rpm_high is not None:
        sec_target_type = {"workoutTargetTypeId": TargetType.CADENCE, "workoutTargetTypeKey": "cadence", "displayOrder": 3}
        sec_one, sec_two = float(rpm_low), float(rpm_high)
    else:
        sec_target_type = sec_one = sec_two = None

    return ExecutableStep(
        stepOrder=step_order,
        stepType={"stepTypeId": st_id, "stepTypeKey": st_key, "displayOrder": st_order},
        endCondition={"conditionTypeId": ConditionType.TIME, "conditionTypeKey": "time", "displayOrder": 2, "displayable": True},
        endConditionValue=float(step_data["duration_sec"]),
        targetType=target_type, targetValueOne=val_one, targetValueTwo=val_two,
        secondaryTargetType=sec_target_type, secondaryTargetValueOne=sec_one, secondaryTargetValueTwo=sec_two,
    )


def build_garmin_workout_payload(workout_name: str, sport: str, workout_text: str, ftp: int) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_workout_structure(workout_text, ftp)
    blocks = parsed["blocks"]
    if not blocks:
        raise ValueError("Geen geldige workout-stappen gevonden (gebruik bijv. '- 15m 65% 90rpm').")

    garmin_steps: list[ExecutableStep | RepeatGroup] = []
    order_counter = 1
    for blk in blocks:
        if blk["type"] == "step":
            garmin_steps.append(_make_executable_step(blk, order_counter))
            order_counter += 1
        elif blk["type"] == "repeat":
            group_order = order_counter
            order_counter += 1
            child_steps: list[ExecutableStep | RepeatGroup] = []
            for s in blk["steps"]:
                child_steps.append(_make_executable_step(s, order_counter))
                order_counter += 1
            garmin_steps.append(
                RepeatGroup(
                    stepOrder=group_order,
                    stepType={"stepTypeId": StepType.REPEAT, "stepTypeKey": "repeat", "displayOrder": 6},
                    numberOfIterations=int(blk["iterations"]),
                    workoutSteps=child_steps,
                    endCondition={"conditionTypeId": ConditionType.ITERATIONS, "conditionTypeKey": "iterations", "displayOrder": 7, "displayable": False},
                    endConditionValue=float(blk["iterations"]),
                )
            )

    sport_norm = sport.strip().lower()
    if sport_norm in ("run", "running", "hardlopen", "lopen"):
        sport_type_dict = {"sportTypeId": SportType.RUNNING, "sportTypeKey": "running", "displayOrder": 1}
        workout_cls = RunningWorkout
    elif sport_norm in ("swim", "swimming", "zwemmen"):
        sport_type_dict = {"sportTypeId": SportType.SWIMMING, "sportTypeKey": "swimming", "displayOrder": 3}
        workout_cls = SwimmingWorkout
    else:
        sport_type_dict = {"sportTypeId": SportType.CYCLING, "sportTypeKey": "cycling", "displayOrder": 2}
        workout_cls = CyclingWorkout

    segment = WorkoutSegment(segmentOrder=1, sportType=sport_type_dict, workoutSteps=garmin_steps)
    workout_obj = workout_cls(
        workoutName=workout_name.strip() or "AI Endurance Workout",
        estimatedDurationInSecs=max(60, int(parsed["total_seconds"])),
        workoutSegments=[segment],
        description=f"Gegenereerd door AI Endurance Suite (FTP {ftp}W, ~{parsed['estimated_tss']} TSS, ~{parsed['estimated_kj']} kJ)",
    )
    return workout_obj.to_dict(), parsed

GARMIN_DIR = Path(__file__).resolve().parent / "garmin-ai"
GARMIN_DATA_FILE = GARMIN_DIR / "garmin" / "data.json"
GARMIN_SYNC_SCRIPT = GARMIN_DIR / "sync_garmin.py"
SETTINGS_FILE = Path(__file__).resolve().parent / ".streamlit" / "user_settings.json"


def laad_opgeslagen_instellingen() -> dict:
    """Laad lokaal opgeslagen koppelgegevens en parameters uit .streamlit/user_settings.json."""
    defaults = {
        "athlete_id": "",
        "intervals_key": "",
        "gemini_key": "",
        "dagen": 42,
        "ftp": 280,
        "gewicht": 75.0,
        "ai_model": "gemini-3-flash-preview",
    }
    try:
        defaults["athlete_id"] = st.secrets.get("ATHLETE_ID", defaults["athlete_id"])
        defaults["intervals_key"] = st.secrets.get("INTERVALS_KEY", defaults["intervals_key"])
        defaults["gemini_key"] = st.secrets.get("GEMINI_KEY", defaults["gemini_key"])
    except Exception:
        pass

    if SETTINGS_FILE.exists():
        try:
            opgeslagen = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(opgeslagen, dict):
                for k, v in opgeslagen.items():
                    if v is not None and v != "":
                        defaults[k] = v
        except Exception:
            pass
    return defaults


def sla_instellingen_op(instellingen: dict) -> None:
    """Sla koppelgegevens en parameters veilig lokaal op met 0600 rechten."""
    SETTINGS_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(SETTINGS_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(instellingen, f, indent=2)


def vraag_gemini_ai(api_key: str, gekozen_model: str, prompt_tekst: str) -> tuple[bool, str, str]:
    """Roep de officiële Gemini REST API aan met automatische model-fallback."""
    kandidaat_modellen = list(
        dict.fromkeys(
            [
                gekozen_model.strip(),
                "gemini-3-flash-preview",
                "gemini-3.1-pro-preview",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            ]
        )
    )
    laatste_fout = ""
    for m_naam in kandidaat_modellen:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_naam}:generateContent"
        try:
            resp = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key.strip(),
                },
                json={
                    "contents": [{"parts": [{"text": prompt_tekst}]}],
                    "generationConfig": {"temperature": 0.4},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates") or []
                if candidates:
                    parts = (candidates[0].get("content") or {}).get("parts") or []
                    tekst = "".join(p.get("text", "") for p in parts).strip()
                    if tekst:
                        return True, tekst, m_naam
                laatste_fout = f"Leeg antwoord van model {m_naam}"
            else:
                try:
                    err_json = resp.json()
                    err_msg = (err_json.get("error") or {}).get("message") or resp.text
                except Exception:
                    err_msg = resp.text
                laatste_fout = f"HTTP {resp.status_code} ({m_naam}): {err_msg}"
                if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg:
                    return False, f"Je Gemini API Key is ongeldig: {err_msg}", m_naam
        except Exception as exc:
            laatste_fout = f"{m_naam}: {exc}"
    return False, laatste_fout, gekozen_model


def voer_garmin_sync_uit(dagen_sync: int = 2, force_details: bool = False) -> tuple[bool, str]:
    """Voer sync_garmin.py uit voor het opgegeven aantal dagen en maak de cache leeg."""
    cmd = [
        sys.executable,
        str(GARMIN_SYNC_SCRIPT),
        "--days",
        str(dagen_sync),
        "--out",
        str(GARMIN_DIR / "garmin"),
    ]
    if force_details:
        cmd.append("--force-details")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        st.cache_data.clear()
        return True, res.stdout
    return False, (res.stderr or res.stdout)


def stuur_workout_naar_garmin(
    naam: str,
    sport: str,
    datum_str: str,
    workout_tekst: str,
    ftp: int,
) -> tuple[bool, dict | str]:
    """Upload een gestructureerde workout naar Garmin Connect en plan hem op datum_str."""
    try:
        import base64
        from garminconnect import Garmin

        b64_tok = os.environ.get("GARMIN_TOKEN_B64", "").strip()
        if not b64_tok:
            try:
                b64_tok = str(st.secrets.get("GARMIN_TOKEN_B64", "")).strip()
            except Exception:
                b64_tok = ""

        api = Garmin()
        tok_dir = Path.home() / ".garminconnect"
        if b64_tok:
            api.login(tokenstore=base64.b64decode(b64_tok).decode("utf-8"))
        elif tok_dir.exists():
            api.login(tokenstore=str(tok_dir))
        else:
            return False, "Geen Garmin Connect sessie-token gevonden (~/.garminconnect of GARMIN_TOKEN_B64 in Streamlit Secrets)."

        payload, parsed = build_garmin_workout_payload(naam, sport, workout_tekst, int(ftp))
        up_res = api.upload_workout(payload)
        w_id = up_res.get("workoutId") if isinstance(up_res, dict) else None
        if not w_id:
            return False, f"Garmin gaf geen workoutId terug: {up_res}"
        if datum_str:
            api.schedule_workout(w_id, datum_str)

        # Werk ook direct lokale data.json bij indien aanwezig
        if GARMIN_DATA_FILE.exists():
            try:
                raw_data = json.loads(GARMIN_DATA_FILE.read_text(encoding="utf-8"))
                sched_list = raw_data.get("scheduled_workouts") or []
                sched_list.append(
                    {
                        "id": w_id,
                        "workout_id": w_id,
                        "date": datum_str,
                        "title": naam,
                        "sport": sport.lower(),
                        "duration_min": parsed["total_minutes"],
                    }
                )
                raw_data["scheduled_workouts"] = sorted(sched_list, key=lambda x: str(x.get("date") or ""))
                GARMIN_DATA_FILE.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            except Exception:
                pass

        st.cache_data.clear()
        return True, {
            "ok": True,
            "workout_id": w_id,
            "workout_name": naam,
            "scheduled_date": datum_str,
            "estimated_minutes": parsed["total_minutes"],
            "estimated_tss": parsed["estimated_tss"],
            "estimated_kj": parsed["estimated_kj"],
        }
    except Exception as exc:
        return False, str(exc)


# --- PAGINA CONFIGURATIE ---
st.set_page_config(page_title="Pro Performance & AI Endurance Suite", layout="wide")
st.title("🚴‍♂️ Pro Performance & AI Endurance Dashboard")

opgeslagen_cfg = laad_opgeslagen_instellingen()

# Lees alvast Garmin fysiologie (indien aanwezig) voor slimme defaults in de zijbalk
garmin_phys_vooraf: dict = {}
if GARMIN_DATA_FILE.exists():
    try:
        garmin_phys_vooraf = (json.loads(GARMIN_DATA_FILE.read_text(encoding="utf-8")) or {}).get("physiology") or {}
    except Exception:
        garmin_phys_vooraf = {}

# --- ZIJBLAD VOOR INSTELLINGEN ---
with st.sidebar:
    st.header("🔑 Koppelgegevens")

    athlete_id = st.text_input(
        "Athlete ID (bijv. i123456)",
        value=opgeslagen_cfg.get("athlete_id", ""),
        help="Optioneel (Intervals.icu). Wordt automatisch lokaal opgeslagen.",
    )
    intervals_key = st.text_input(
        "Intervals.icu API Key",
        value=opgeslagen_cfg.get("intervals_key", ""),
        type="password",
        help="Optioneel. Wordt automatisch lokaal op je Mac bewaard.",
    )
    gemini_key = st.text_input(
        "Gemini API Key",
        value=opgeslagen_cfg.get("gemini_key", ""),
        type="password",
        help="Wordt automatisch lokaal op je Mac bewaard zodat je hem maar één keer hoeft in te vullen.",
    )

    if gemini_key or intervals_key or athlete_id:
        st.caption("✅ Koppelgegevens automatisch lokaal opgeslagen")

    st.divider()
    st.header("⚙️ Atletische Parameters")
    dagen = st.slider(
        "Aantal dagen historie",
        min_value=14,
        max_value=180,
        value=int(opgeslagen_cfg.get("dagen", 42)),
        help="Bepaalt de periode voor grafieken, HRV-baselines en AI-context.",
    )

    default_ftp = int(opgeslagen_cfg.get("ftp") or garmin_phys_vooraf.get("cycling_ftp_w") or 280)
    default_gewicht = float(opgeslagen_cfg.get("gewicht") or garmin_phys_vooraf.get("weight_kg") or 75.0)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        handmatig_ftp = st.number_input(
            "FTP (Watt)",
            min_value=100,
            max_value=500,
            value=default_ftp,
            step=5,
            help="Wordt gebruikt voor vermogenszones, TSS, IF en Garmin Workout targets.",
        )
    with col_s2:
        handmatig_gewicht = st.number_input(
            "Gewicht (kg)",
            min_value=40.0,
            max_value=130.0,
            value=default_gewicht,
            step=0.5,
            help="Wordt gebruikt voor W/kg en voedingsberekeningen.",
        )

    standaard_modellen = [
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.8-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    opgeslagen_model = opgeslagen_cfg.get("ai_model", "gemini-3-flash-preview")
    if opgeslagen_model not in standaard_modellen:
        standaard_modellen = [opgeslagen_model] + standaard_modellen

    ai_model = st.selectbox(
        "AI Modelnaam",
        options=standaard_modellen,
        index=standaard_modellen.index(opgeslagen_model),
        accept_new_options=True,
        help="Kies een actueel Gemini 3 model of typ zelf een modelnaam in.",
    )

    # Sla wijzigingen direct automatisch op
    huidige_cfg = {
        "athlete_id": athlete_id.strip(),
        "intervals_key": intervals_key.strip(),
        "gemini_key": gemini_key.strip(),
        "dagen": int(dagen),
        "ftp": int(handmatig_ftp),
        "gewicht": float(handmatig_gewicht),
        "ai_model": ai_model,
    }
    if huidige_cfg != opgeslagen_cfg:
        sla_instellingen_op(huidige_cfg)

    if st.button("Data verversen", icon=":material/refresh:", width="stretch"):
        st.cache_data.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("⌚ Garmin Connect (Lokaal)")
    if GARMIN_DATA_FILE.exists():
        try:
            g_meta = json.loads(GARMIN_DATA_FILE.read_text(encoding="utf-8"))
            st.caption(f"Laatste sync: `{g_meta.get('updated_at', 'onbekend')}`")
        except Exception:
            st.caption("Lokale `garmin/data.json` aanwezig.")
    else:
        st.caption("Nog geen lokale Garmin data gevonden.")

    if st.button("Hersync laatste workout & analyseer", icon=":material/bolt:", type="primary", width="stretch"):
        with st.spinner("Laatste Garmin workout, splits & hersteldata ophalen..."):
            ok, msg = voer_garmin_sync_uit(2, force_details=True)
            if ok:
                st.session_state.trigger_laatste_workout_analyse = True
                st.toast("Laatste Garmin workout gesynchroniseerd!", icon=":material/check_circle:")
                st.rerun()
            else:
                st.error(f"Garmin sync mislukt: {msg}")

    if st.button(f"Volledige Garmin sync ({dagen}d)", icon=":material/sync:", width="stretch"):
        with st.spinner(f"Garmin data, laps & zones van afgelopen {dagen} dagen ophalen..."):
            ok, msg = voer_garmin_sync_uit(dagen)
            if ok:
                st.toast("Volledige Garmin historie gesynchroniseerd!", icon=":material/check_circle:")
                st.rerun()
            else:
                st.error(f"Garmin sync mislukt: {msg}")


def laad_lokale_garmin_data(dagen_terug: int, ftp_referentie: int):
    """
    Laad lokale garmin-ai/garmin/data.json en zet om naar:
    df_act, df_wel (inclusief lokaal berekende CTL/ATL/TSB), physiology_dict, scheduled_list.
    """
    if not GARMIN_DATA_FILE.exists():
        return pd.DataFrame(), pd.DataFrame(), {}, []
    try:
        raw = json.loads(GARMIN_DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    grens_datum = (datetime.today() - timedelta(days=dagen_terug)).strftime("%Y-%m-%d")
    physiology = raw.get("physiology") or {}
    scheduled_workouts = raw.get("scheduled_workouts") or []

    # Verwerk alle activiteiten (ook iets oudere indien aanwezig voor nauwkeurigere CTL-opbouw)
    act_dict = raw.get("activities") or {}
    all_act_rows = []
    for _, a in act_dict.items():
        if not isinstance(a, dict) or not a.get("date"):
            continue
        duur_min = float(a.get("duration_minutes") or 0.0)
        duur_sec = duur_min * 60.0
        duur_uur = round(duur_min / 60.0, 2)
        avg_w = a.get("avg_power_w")
        np_w = a.get("norm_power_w") or avg_w
        avg_hr = a.get("avg_hr_bpm")
        arbeid_kj = round((avg_w * duur_sec) / 1000.0, 0) if avg_w and duur_sec else 0.0

        # Bereken Intensity Factor (IF) & Power TSS indien niet of deels aanwezig
        if_val = round((np_w / ftp_referentie) * 100.0, 1) if (np_w and ftp_referentie > 0) else None
        power_tss = (
            round(((duur_sec * np_w * (np_w / ftp_referentie)) / (ftp_referentie * 3600.0)) * 100.0, 1)
            if (np_w and ftp_referentie > 0 and duur_sec > 0)
            else None
        )
        garmin_load = a.get("training_load")
        # Gebruik Power TSS voor wielrennen met vermogensmeter, of Garmin's EPOC Training Load als fallback
        tss_load = power_tss if power_tss is not None else (float(garmin_load) if garmin_load is not None else 0.0)

        ef = a.get("efficiency_factor")
        if ef is None and np_w and avg_hr and avg_hr > 40:
            ef = round(np_w / avg_hr, 2)

        pz = a.get("power_zones") or []
        hrz = a.get("hr_zones") or []
        zone_times = pz if any(z.get("secs", 0) > 0 for z in pz if isinstance(z, dict)) else hrz

        all_act_rows.append(
            {
                "id": a.get("id"),
                "datum": str(a.get("date")),
                "start_date_local": a.get("start_time_local") or a.get("date"),
                "name": a.get("name"),
                "type": a.get("type"),
                "duur_min": round(duur_min, 0),
                "duur_uur": duur_uur,
                "afstand_km": a.get("distance_km") or 0.0,
                "average_heartrate": avg_hr,
                "max_heartrate": a.get("max_hr_bpm"),
                "average_watts": avg_w,
                "weighted_average_watts": a.get("norm_power_w"),
                "icu_training_load": round(tss_load, 1),
                "garmin_training_load": garmin_load,
                "icu_intensity": if_val,
                "icu_efficiency_factor": ef,
                "decoupling": a.get("decoupling_pct"),
                "arbeid_kj": arbeid_kj,
                "aerobic_te": a.get("aerobic_te"),
                "anaerobic_te": a.get("anaerobic_te"),
                "calories": a.get("calories"),
                "icu_zone_times": zone_times,
                "power_zones": pz,
                "hr_zones": hrz,
                "laps": a.get("laps") or [],
            }
        )

    df_all_act = pd.DataFrame(all_act_rows).sort_values("start_date_local") if all_act_rows else pd.DataFrame()
    df_g_act = df_all_act[df_all_act["datum"] >= grens_datum].copy() if not df_all_act.empty else pd.DataFrame()

    # Wellness + Lokale PMC (CTL / ATL / TSB) berekening
    wel_dict = raw.get("wellness") or {}
    wel_rows = []
    for datum, w in wel_dict.items():
        if isinstance(w, dict):
            wel_rows.append(
                {
                    "datum": str(datum),
                    "restingHR": w.get("resting_hr_bpm"),
                    "hrv": w.get("hrv_overnight_ms"),
                    "hrv_status": w.get("hrv_status"),
                    "slaap_uren": w.get("sleep_hours"),
                    "sleepScore": w.get("sleep_score"),
                    "body_battery_low": w.get("body_battery_low"),
                    "body_battery_high": w.get("body_battery_high"),
                    "stress_avg": w.get("stress_avg"),
                    "steps": w.get("steps"),
                    "training_readiness": w.get("training_readiness"),
                    "training_readiness_level": w.get("training_readiness_level"),
                    "weight": w.get("weight_kg"),
                }
            )
    df_all_wel = pd.DataFrame(wel_rows).sort_values("datum") if wel_rows else pd.DataFrame()

    # Bereken dagelijkse EWMA CTL (42d), ATL (7d) en TSB = CTL - ATL over alle beschikbare dagen
    alle_datums = []
    if not df_all_wel.empty:
        alle_datums.extend(df_all_wel["datum"].dropna().tolist())
    if not df_all_act.empty:
        alle_datums.extend(df_all_act["datum"].dropna().tolist())

    if alle_datums:
        min_dt = pd.to_datetime(min(alle_datums))
        max_dt = pd.to_datetime(max( max(alle_datums), datetime.today().strftime("%Y-%m-%d") ))
        kalender = pd.DataFrame({"datum": pd.date_range(min_dt, max_dt, freq="D").strftime("%Y-%m-%d")})

        if not df_all_act.empty:
            dag_tss = df_all_act.groupby("datum")["icu_training_load"].sum().reset_index()
            kalender = kalender.merge(dag_tss, on="datum", how="left")
            kalender["icu_training_load"] = kalender["icu_training_load"].fillna(0.0)
        else:
            kalender["icu_training_load"] = 0.0

        # Start CTL/ATL op basis van Garmin's eigen Chronic/Acute load (indien gerapporteerd) of het gemiddelde van de eerste week
        eerste_gem = float(kalender["icu_training_load"].head(14).mean()) if not kalender.empty else 30.0
        ctl_val = max(20.0, eerste_gem)
        atl_val = max(20.0, eerste_gem)
        ctl_list, atl_list, tsb_list = [], [], []
        for tss in kalender["icu_training_load"]:
            tsb_list.append(round(ctl_val - atl_val, 1))
            ctl_val = ctl_val + (float(tss) - ctl_val) / 42.0
            atl_val = atl_val + (float(tss) - atl_val) / 7.0
            ctl_list.append(round(ctl_val, 1))
            atl_list.append(round(atl_val, 1))

        kalender["ctl"] = ctl_list
        kalender["atl"] = atl_list
        kalender["tsb"] = tsb_list

        if not df_all_wel.empty:
            df_all_wel = pd.merge(kalender[["datum", "ctl", "atl", "tsb"]], df_all_wel, on="datum", how="left")
        else:
            df_all_wel = kalender[["datum", "ctl", "atl", "tsb"]].copy()

    df_g_wel = df_all_wel[df_all_wel["datum"] >= grens_datum].copy() if not df_all_wel.empty else pd.DataFrame()
    return df_g_act, df_g_wel, physiology, scheduled_workouts


# --- DATA OPHALEN & VERWERKEN (INTERVALS.ICU) ---
@st.cache_data(ttl=600)
def haal_data_op(athlete_id, api_key, dagen_terug):
    vandaag = datetime.today()
    eind_datum = vandaag.strftime("%Y-%m-%d")
    start_datum = (vandaag - timedelta(days=dagen_terug)).strftime("%Y-%m-%d")
    toekomst_datum = (vandaag + timedelta(days=7)).strftime("%Y-%m-%d")
    auth = ("API_KEY", api_key)

    act_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities?oldest={start_datum}&newest={eind_datum}"
    wel_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness?oldest={start_datum}&newest={eind_datum}"
    evt_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/events?oldest={eind_datum}&newest={toekomst_datum}"

    act_res = requests.get(act_url, auth=auth)
    wel_res = requests.get(wel_url, auth=auth)
    evt_res = requests.get(evt_url, auth=auth)

    if act_res.status_code != 200 or wel_res.status_code != 200:
        return None, None, None

    df_act = pd.DataFrame(act_res.json())
    df_wel = pd.DataFrame(wel_res.json())
    df_evt = pd.DataFrame(evt_res.json()) if evt_res.status_code == 200 else pd.DataFrame()

    if not df_act.empty:
        if "type" in df_act.columns:
            df_act = df_act[df_act["type"].notna()].copy()
        if "start_date_local" in df_act.columns:
            df_act["datum"] = pd.to_datetime(df_act["start_date_local"]).dt.strftime("%Y-%m-%d")
            df_act = df_act.sort_values("start_date_local")
        if "moving_time" in df_act.columns:
            df_act["duur_min"] = (df_act["moving_time"] / 60).round(0)
            df_act["duur_uur"] = (df_act["moving_time"] / 3600).round(2)
        if "distance" in df_act.columns:
            df_act["afstand_km"] = (df_act["distance"] / 1000).round(1)
        if "icu_joules" in df_act.columns:
            df_act["arbeid_kj"] = (df_act["icu_joules"] / 1000).round(0)
        elif "average_watts" in df_act.columns and "moving_time" in df_act.columns:
            df_act["arbeid_kj"] = ((df_act["average_watts"] * df_act["moving_time"]) / 1000).round(0)
        else:
            df_act["arbeid_kj"] = 0

    if not df_wel.empty and "id" in df_wel.columns:
        df_wel["datum"] = pd.to_datetime(df_wel["id"]).dt.strftime("%Y-%m-%d")
        df_wel = df_wel.sort_values("datum")
        if "sleepSecs" in df_wel.columns:
            df_wel["slaap_uren"] = (df_wel["sleepSecs"] / 3600).round(1)
        if "ctl" in df_wel.columns and "atl" in df_wel.columns:
            df_wel["ctl"] = df_wel["ctl"].round(1)
            df_wel["atl"] = df_wel["atl"].round(1)
            df_wel["tsb"] = (df_wel["ctl"] - df_wel["atl"]).round(1)

    return df_act, df_wel, df_evt


# --- HULPFUNCTIE: BEREKEN READINESS SCORE ---
def bereken_readiness(df_wel):
    if df_wel.empty:
        return 50, "Onvoldoende data", "Geen recente wellness-metrics gevonden."

    score = 70.0
    redenen = []

    garmin_tr = None
    if "training_readiness" in df_wel.columns and df_wel["training_readiness"].notna().any():
        garmin_tr = float(df_wel["training_readiness"].dropna().iloc[-1])
        score = garmin_tr
        redenen.append(f"Garmin Readiness ({garmin_tr:.0f}/100)")

    if "hrv" in df_wel.columns and df_wel["hrv"].dropna().shape[0] >= 3:
        hrv_reeks = df_wel["hrv"].dropna()
        laatste_hrv = hrv_reeks.iloc[-1]
        gem_hrv = hrv_reeks.mean()
        std_hrv = hrv_reeks.std() if hrv_reeks.std() > 0 else 1
        z_hrv = (laatste_hrv - gem_hrv) / std_hrv

        if garmin_tr is None:
            hrv_impact = np.clip(z_hrv * 12, -25, 15)
            score += hrv_impact
        redenen.append(f"HRV ({laatste_hrv:.0f} ms vs. {gem_hrv:.0f} ms baseline)")

    if "restingHR" in df_wel.columns and df_wel["restingHR"].dropna().shape[0] >= 3:
        rhr_reeks = df_wel["restingHR"].dropna()
        laatste_rhr = rhr_reeks.iloc[-1]
        gem_rhr = rhr_reeks.mean()
        verschil_rhr = gem_rhr - laatste_rhr

        if garmin_tr is None:
            rhr_impact = np.clip(verschil_rhr * 2.5, -15, 10)
            score += rhr_impact
        redenen.append(f"RHR ({laatste_rhr:.0f} bpm vs. {gem_rhr:.0f} bpm gem)")

    if "slaap_uren" in df_wel.columns and df_wel["slaap_uren"].notna().any():
        laatste_slaap = df_wel["slaap_uren"].dropna().iloc[-1]
        if garmin_tr is None:
            slaap_impact = np.clip((laatste_slaap - 7.5) * 6, -15, 10)
            score += slaap_impact
        redenen.append(f"Slaap ({laatste_slaap:.1f}u)")

    if "body_battery_high" in df_wel.columns and df_wel["body_battery_high"].notna().any():
        laatste_bb = df_wel["body_battery_high"].dropna().iloc[-1]
        redenen.append(f"Body Battery (max {laatste_bb:.0f})")

    if "tsb" in df_wel.columns and df_wel["tsb"].notna().any():
        laatste_tsb = df_wel["tsb"].dropna().iloc[-1]
        if garmin_tr is None:
            if laatste_tsb < -20:
                score -= 12
            elif laatste_tsb > 0:
                score += 5

    eind_score = int(np.clip(score, 10, 100))
    if eind_score >= 75:
        advies = "🟢 Groen Licht: Hoog herstel. Ideaal voor VO2max, drempelintervallen of een lange duurrit."
    elif eind_score >= 50:
        advies = "🟡 Oranje Licht: Normale belasting. Focus op Zone 2 duur of gecontroleerde tempo-blokken."
    else:
        advies = "🔴 Rood Licht: Verhoogde vermoeidheid. Kies voor een rustdag of maximaal 45-60 min Zone 1 losrijden."

    return eind_score, advies, " | ".join(redenen)


# --- HOOFDSCHERM ---
df_g_act, df_g_wel, garmin_phys, garmin_scheduled = laad_lokale_garmin_data(dagen, handmatig_ftp)
heeft_garmin = not df_g_wel.empty or not df_g_act.empty
heeft_intervals = bool(athlete_id and intervals_key)

if heeft_intervals or heeft_garmin:
    if heeft_intervals:
        df_act, df_wel, df_evt = haal_data_op(athlete_id, intervals_key, dagen)
        if df_act is None:
            st.warning("⚠️ Kon Intervals.icu niet bereiken; schakel over op lokale Garmin data.")
            df_act, df_wel, df_evt = df_g_act, df_g_wel, pd.DataFrame()
        else:
            if not df_g_act.empty:
                if df_act.empty:
                    df_act = df_g_act.copy()
                else:
                    bestaande_datums = set(df_act["datum"].dropna().unique()) if "datum" in df_act.columns else set()
                    nieuwe_g_act = df_g_act[~df_g_act["datum"].isin(bestaande_datums)]
                    if not nieuwe_g_act.empty:
                        df_act = pd.concat([df_act, nieuwe_g_act], ignore_index=True).sort_values("datum")

            if not df_g_wel.empty:
                garmin_extra_cols = [
                    c for c in [
                        "datum", "training_readiness", "training_readiness_level",
                        "body_battery_low", "body_battery_high", "stress_avg", "steps",
                        "restingHR", "hrv", "slaap_uren", "sleepScore", "weight", "ctl", "atl", "tsb"
                    ] if c in df_g_wel.columns
                ]
                df_wel = pd.merge(
                    df_wel,
                    df_g_wel[garmin_extra_cols],
                    on="datum",
                    how="outer",
                    suffixes=("", "_garmin"),
                ).sort_values("datum")
                for base_col in ["restingHR", "hrv", "slaap_uren", "sleepScore", "weight", "ctl", "atl", "tsb"]:
                    g_col = f"{base_col}_garmin"
                    if g_col in df_wel.columns:
                        if base_col in df_wel.columns:
                            df_wel[base_col] = df_wel[base_col].combine_first(df_wel[g_col])
                        else:
                            df_wel[base_col] = df_wel[g_col]
                        df_wel.drop(columns=[g_col], inplace=True)
    else:
        df_act, df_wel, df_evt = df_g_act, df_g_wel, pd.DataFrame()

    # Bepaal laatst bekende gewicht en FTP uit Garmin/Intervals, of gebruik sidebar waarde
    actueel_gewicht = handmatig_gewicht
    if not df_wel.empty and "weight" in df_wel.columns and df_wel["weight"].notna().any():
        actueel_gewicht = float(df_wel["weight"].dropna().iloc[-1])
    elif garmin_phys.get("weight_kg"):
        actueel_gewicht = float(garmin_phys["weight_kg"])

    actueel_ftp = handmatig_ftp
    if not df_act.empty and "icu_ftp" in df_act.columns and df_act["icu_ftp"].notna().any():
        actueel_ftp = int(df_act["icu_ftp"].dropna().iloc[-1])
    elif garmin_phys.get("cycling_ftp_w"):
        actueel_ftp = int(garmin_phys["cycling_ftp_w"])

    # --- DAILY READINESS BANNER & POST-WORKOUT SNEL-SYNC ---
    readiness_score, readiness_advies, readiness_details = bereken_readiness(df_wel)
    col_r1, col_r2, col_r3 = st.columns([1.1, 2.7, 1.6])
    with col_r1:
        st.metric(
            "🔋 Daily Readiness",
            f"{readiness_score} / 100",
            border=True,
            help="Gecombineerde score uit je Garmin Training Readiness, HRV-trend, rusthartslag, slaap en TSB.",
        )
    with col_r2:
        st.info(f"**{readiness_advies}** \n*Onderbouwing laatste meting: {readiness_details}*")
    with col_r3:
        with st.container(border=True):
            if not df_act.empty:
                laatste_rit = df_act.iloc[-1]
                rit_naam = laatste_rit.get("name", "Workout")
                rit_datum = laatste_rit.get("datum", "")
                st.caption(f"Laatste workout: **{rit_naam}** ({rit_datum})")
            else:
                st.caption("Klaar met je training?")

            if st.button(
                "Hersync Garmin workout & analyseer",
                icon=":material/bolt:",
                type="primary",
                width="stretch",
                key="top_sync_and_analyze",
            ):
                with st.spinner("Nieuwste Garmin workout, laps & zones ophalen..."):
                    ok, msg = voer_garmin_sync_uit(2, force_details=True)
                    if ok:
                        st.session_state.trigger_laatste_workout_analyse = True
                        st.toast("Garmin workout gesynchroniseerd!", icon=":material/check_circle:")
                        st.rerun()
                    else:
                        st.error(f"Sync mislukt: {msg}")

    # --- 1. TOP KPI'S ---
    st.subheader(f"⚡ Kerncijfers afgelopen {dagen} dagen")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        uren = round(df_act["duur_uur"].sum(), 1) if not df_act.empty and "duur_uur" in df_act.columns else 0
        aantal = len(df_act) if not df_act.empty else 0
        st.metric("Volume", f"{uren} uur", f"{aantal} sessies")
    with k2:
        km = round(df_act["afstand_km"].sum(), 0) if not df_act.empty and "afstand_km" in df_act.columns else 0
        kj_tot = int(df_act["arbeid_kj"].sum()) if not df_act.empty and "arbeid_kj" in df_act.columns else 0
        st.metric("Afstand & Arbeid", f"{int(km)} km", f"{kj_tot:,} kJ verbrand")
    with k3:
        load = int(df_act["icu_training_load"].sum()) if not df_act.empty and "icu_training_load" in df_act.columns else 0
        gem_load = int(load / (dagen / 7)) if dagen > 0 else 0
        st.metric("Totale Load (TSS)", f"{load}", f"~{gem_load} TSS/week")
    with k4:
        if not df_wel.empty and "ctl" in df_wel.columns and df_wel["ctl"].notna().any():
            huidige_ctl = round(df_wel["ctl"].dropna().iloc[-1], 1)
            start_ctl = round(df_wel["ctl"].dropna().iloc[0], 1)
            verschil_ctl = round(huidige_ctl - start_ctl, 1)
            st.metric("Fitness (CTL)", huidige_ctl, f"{verschil_ctl:+} in periode")
        else:
            st.metric("Fitness (CTL)", "-")
    with k5:
        if not df_wel.empty and "tsb" in df_wel.columns and df_wel["tsb"].notna().any():
            huidige_tsb = round(df_wel["tsb"].dropna().iloc[-1], 1)
            status = "Fris (Race klaar)" if huidige_tsb > 5 else ("Optimale opbouw" if huidige_tsb >= -15 else "Zwaar belast")
            st.metric("Vorm (TSB)", huidige_tsb, status)
        else:
            st.metric("Vorm (TSB)", "-")
    with k6:
        w_kg = round(actueel_ftp / actueel_gewicht, 2) if actueel_gewicht > 0 else 0
        vo2_badge = garmin_phys.get("vo2max_cycling") or garmin_phys.get("vo2max_running")
        sub_label = f"FTP {actueel_ftp}W | VO2max {vo2_badge}" if vo2_badge else f"FTP: {actueel_ftp}W @ {actueel_gewicht}kg"
        st.metric("Relatief Vermogen", f"{w_kg} W/kg", sub_label)

    st.divider()

    # --- 2. UITGEBREIDE TABS MET AI ENDURANCE FEATURES ---
    tab_pmc, tab_herstel, tab_power, tab_durability, tab_voeding, tab_predict, tab_agenda, tab_data = st.tabs([
        "📈 PMC & Belasting",
        "💤 Herstel & Garmin Wellness",
        "⚡ Vermogen, Zones & Laps",
        "🛡️ Durability (kJ)",
        "🍝 Voeding & Macros",
        "🔮 Fysiologie & Race Predictor",
        "💬 AI Workout Chat & Garmin Export",
        "📋 Ruwe Data",
    ])

    # TAB 1: PMC
    with tab_pmc:
        if not df_wel.empty and "ctl" in df_wel.columns:
            fig_pmc = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                subplot_titles=("Fitness (CTL) vs. Vermoeidheid (ATL) & Dagelijkse TSS", "Vorm / Freshness (TSB = CTL - ATL)"),
            )

            if not df_act.empty and "icu_training_load" in df_act.columns:
                dag_load = df_act.groupby("datum").agg(
                    totale_tss=("icu_training_load", "sum"),
                    ritten=("name", lambda x: ", ".join(x.dropna())),
                ).reset_index()

                fig_pmc.add_trace(go.Bar(
                    x=dag_load["datum"], y=dag_load["totale_tss"], name="Dagelijkse TSS",
                    marker_color="rgba(150, 150, 150, 0.4)", customdata=dag_load["ritten"],
                    hovertemplate="<b>Load: %{y:.0f} TSS</b><br>%{customdata}<extra></extra>",
                ), row=1, col=1)

            fig_pmc.add_trace(go.Scatter(x=df_wel["datum"], y=df_wel["ctl"], name="Fitness (CTL)", line=dict(color="#00b4d8", width=3)), row=1, col=1)
            fig_pmc.add_trace(go.Scatter(x=df_wel["datum"], y=df_wel["atl"], name="Vermoeidheid (ATL)", line=dict(color="#ef476f", width=2, dash="dot")), row=1, col=1)

            if "tsb" in df_wel.columns:
                kleuren_tsb = ["#06d6a0" if val >= 0 else "#ffd166" if val >= -15 else "#ef476f" for val in df_wel["tsb"].fillna(0)]
                fig_pmc.add_trace(go.Bar(x=df_wel["datum"], y=df_wel["tsb"], name="Vorm (TSB)", marker_color=kleuren_tsb), row=2, col=1)
                fig_pmc.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                fig_pmc.add_hline(y=-15, line_dash="dot", line_color="#ef476f", row=2, col=1)

            fig_pmc.update_layout(height=550, hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig_pmc)
        else:
            st.info("Geen CTL/ATL of trainingsbelasting data beschikbaar.")

    # TAB 2: HERSTEL
    with tab_herstel:
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            if not df_wel.empty and ("hrv" in df_wel.columns or "restingHR" in df_wel.columns):
                fig_hrv = make_subplots(specs=[[{"secondary_y": True}]])
                if "hrv" in df_wel.columns:
                    df_wel["hrv_7d"] = df_wel["hrv"].rolling(7, min_periods=1).mean()
                    fig_hrv.add_trace(go.Scatter(x=df_wel["datum"], y=df_wel["hrv"], name="HRV Dagelijks (ms)", mode="markers", marker=dict(color="#06d6a0", size=7, opacity=0.6)), secondary_y=False)
                    fig_hrv.add_trace(go.Scatter(x=df_wel["datum"], y=df_wel["hrv_7d"], name="HRV 7-daags Gem.", mode="lines", line=dict(color="#06d6a0", width=3)), secondary_y=False)
                if "restingHR" in df_wel.columns:
                    fig_hrv.add_trace(go.Scatter(x=df_wel["datum"], y=df_wel["restingHR"], name="Rusthartslag (bpm)", mode="lines+markers", line=dict(color="#ef476f", width=2)), secondary_y=True)
                fig_hrv.update_layout(title="Autonoom Zenuwstelsel: HRV Baseline vs. Rusthartslag", hovermode="x unified", height=420, template="plotly_dark")
                st.plotly_chart(fig_hrv)

        with col_h2:
            if not df_wel.empty and "slaap_uren" in df_wel.columns:
                fig_sleep = make_subplots(specs=[[{"secondary_y": True}]])
                fig_sleep.add_trace(go.Bar(x=df_wel["datum"], y=df_wel["slaap_uren"], name="Slaapduur (u)", marker_color="#118ab2", opacity=0.7), secondary_y=False)
                if "sleepScore" in df_wel.columns:
                    fig_sleep.add_trace(go.Scatter(x=df_wel["datum"], y=df_wel["sleepScore"], name="Slaap Score", mode="lines+markers", line=dict(color="#ffd166", width=3)), secondary_y=True)
                fig_sleep.add_hline(y=7.5, line_dash="dot", line_color="#06d6a0", annotation_text="7.5u Richtlijn", secondary_y=False)
                fig_sleep.update_layout(title="Slaapduur & Kwaliteit", hovermode="x unified", height=420, template="plotly_dark")
                st.plotly_chart(fig_sleep)

        if not df_wel.empty and any(c in df_wel.columns for c in ["training_readiness", "body_battery_high", "stress_avg"]):
            col_h3, col_h4 = st.columns(2)
            with col_h3:
                if "training_readiness" in df_wel.columns or "body_battery_high" in df_wel.columns:
                    fig_bb = go.Figure()
                    if "body_battery_high" in df_wel.columns:
                        fig_bb.add_trace(go.Bar(x=df_wel["datum"], y=df_wel["body_battery_high"], name="Body Battery Max", marker_color="#06d6a0", opacity=0.65))
                    if "body_battery_low" in df_wel.columns:
                        fig_bb.add_trace(go.Bar(x=df_wel["datum"], y=df_wel["body_battery_low"], name="Body Battery Min", marker_color="#118ab2", opacity=0.85))
                    if "training_readiness" in df_wel.columns:
                        fig_bb.add_trace(go.Scatter(x=df_wel["datum"], y=df_wel["training_readiness"], name="Training Readiness", mode="lines+markers", line=dict(color="#ffd166", width=3)))
                    fig_bb.update_layout(title="Garmin Training Readiness & Body Battery", barmode="overlay", hovermode="x unified", height=380, template="plotly_dark")
                    st.plotly_chart(fig_bb)
            with col_h4:
                if "stress_avg" in df_wel.columns:
                    fig_str = go.Figure()
                    fig_str.add_trace(go.Bar(x=df_wel["datum"], y=df_wel["stress_avg"], name="Gemiddelde Stress", marker_color="#ef476f", opacity=0.75))
                    fig_str.update_layout(title="Garmin Dagelijkse Stress (Gemiddeld)", hovermode="x unified", height=380, template="plotly_dark")
                    st.plotly_chart(fig_str)

    # TAB 3: VERMOGEN, 80/20 POLARISATIE ZONES & WORKOUT LAP INSPECTOR
    with tab_power:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            df_power = df_act[df_act["average_watts"].notna()].copy() if not df_act.empty and "average_watts" in df_act.columns else pd.DataFrame()
            if not df_power.empty:
                fig_pow = go.Figure()
                fig_pow.add_trace(go.Bar(x=df_power["datum"], y=df_power["average_watts"], name="Gem. Wattage", marker_color="#118ab2", customdata=df_power["name"], hovertemplate="<b>%{customdata}</b><br>Gem: %{y} W<extra></extra>"))
                if "weighted_average_watts" in df_power.columns:
                    fig_pow.add_trace(go.Scatter(x=df_power["datum"], y=df_power["weighted_average_watts"], name="Normalized Power (NP)", mode="markers+lines", line=dict(color="#ffd166", width=2)))
                fig_pow.update_layout(title="Geleverde Wattages per Rit (Avg vs. NP)", hovermode="x unified", height=400, template="plotly_dark")
                st.plotly_chart(fig_pow)
            else:
                st.info("Geen vermogensdata gevonden.")

        with col_p2:
            zone_bron = st.radio(
                "Zone-bron voor 80/20 Polarisatie:",
                ["⚡ Vermogenszones (Power)", "❤️ Hartslagzones (HR)"],
                horizontal=True,
                label_visibility="collapsed",
            )
            kolom_keuze = "power_zones" if "Vermogen" in zone_bron else "hr_zones"
            if not df_act.empty:
                zone_totals = {}
                bron_kolom = kolom_keuze if kolom_keuze in df_act.columns else ("icu_zone_times" if "icu_zone_times" in df_act.columns else None)
                if bron_kolom:
                    for z_list in df_act[bron_kolom].dropna():
                        if isinstance(z_list, list):
                            for z in z_list:
                                if isinstance(z, dict) and "id" in z and "secs" in z and float(z["secs"]) > 0:
                                    zone_totals[z["id"]] = zone_totals.get(z["id"], 0.0) + (float(z["secs"]) / 60.0)
                if zone_totals:
                    df_z = pd.DataFrame(list(zone_totals.items()), columns=["Zone", "Minuten"])
                    df_z["Minuten"] = df_z["Minuten"].round(1)
                    totaal_min = df_z["Minuten"].sum()
                    df_z["Percentage"] = (df_z["Minuten"] / totaal_min * 100).round(1) if totaal_min > 0 else 0.0

                    laag_pct = df_z[df_z["Zone"].isin(["Z1", "Z2"])]["Percentage"].sum()
                    hoog_pct = df_z[~df_z["Zone"].isin(["Z1", "Z2"])]["Percentage"].sum()

                    fig_z = px.bar(
                        df_z, x="Zone", y="Percentage", text="Percentage",
                        title=f"Tijd in Zones ({laag_pct:.0f}% Laag Z1-Z2 / {hoog_pct:.0f}% Intensief Z3+)",
                        color="Zone", color_discrete_sequence=px.colors.sequential.Tealgrn,
                        hover_data=["Minuten"],
                    )
                    fig_z.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_z.update_layout(height=360, template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig_z)
                else:
                    st.info("Klik links op **Volledige Garmin sync** om alle vermogens- en hartslagzones per rit in te laden.")

        # --- NIEUW: WORKOUT LAP & INTERVAL INSPECTOR ---
        st.divider()
        st.subheader("🔍 Garmin Workout Lap & Interval Inspector")
        if not df_act.empty and "laps" in df_act.columns:
            ritten_met_laps = df_act[df_act["laps"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
            if not ritten_met_laps.empty:
                ritten_met_laps["label"] = (
                    ritten_met_laps["datum"].astype(str)
                    + " — "
                    + ritten_met_laps["name"].astype(str)
                    + " ("
                    + ritten_met_laps["laps"].apply(len).astype(str)
                    + " laps/intervallen)"
                )
                opties = ritten_met_laps["label"].tolist()[::-1]
                gekozen_label = st.selectbox("Selecteer een workout om de exacte intervallen/laps te inspecteren:", opties)
                gekozen_rit = ritten_met_laps[ritten_met_laps["label"] == gekozen_label].iloc[0]
                df_laps = pd.DataFrame(gekozen_rit["laps"])

                col_l1, col_l2 = st.columns([1.3, 1.2])
                with col_l1:
                    fig_laps = make_subplots(specs=[[{"secondary_y": True}]])
                    if "avg_power_w" in df_laps.columns and df_laps["avg_power_w"].notna().any():
                        fig_laps.add_trace(
                            go.Bar(
                                x=df_laps["lap"].apply(lambda x: f"Lap {x}"),
                                y=df_laps["avg_power_w"],
                                name="Gem. Vermogen (W)",
                                marker_color="#00b4d8",
                                text=df_laps["avg_power_w"],
                                textposition="auto",
                            ),
                            secondary_y=False,
                        )
                    if "avg_hr_bpm" in df_laps.columns and df_laps["avg_hr_bpm"].notna().any():
                        fig_laps.add_trace(
                            go.Scatter(
                                x=df_laps["lap"].apply(lambda x: f"Lap {x}"),
                                y=df_laps["avg_hr_bpm"],
                                name="Gem. Hartslag (bpm)",
                                mode="lines+markers",
                                line=dict(color="#ef476f", width=3),
                            ),
                            secondary_y=True,
                        )
                    fig_laps.update_layout(
                        title=f"Vermogen & Hartslag per Lap/Interval — {gekozen_rit['name']}",
                        height=360,
                        template="plotly_dark",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_laps)
                with col_l2:
                    st.dataframe(
                        df_laps,
                        column_config={
                            "lap": "Lap #",
                            "duration_min": "Duur (min)",
                            "distance_km": "Afstand (km)",
                            "avg_power_w": "Gem. W",
                            "norm_power_w": "NP (W)",
                            "max_power_w": "Max W",
                            "avg_hr_bpm": "Gem. HR",
                            "max_hr_bpm": "Max HR",
                            "avg_cadence_rpm": "Cadans",
                            "efficiency_factor": "EF (W/HR)",
                        },
                        hide_index=True,
                    )
            else:
                st.info("Nog geen lap-splits geladen. Klik op **Hersync laatste workout & analyseer** om de interval-splits op te halen.")

    # TAB 4: DURABILITY (VERMOEIDHEIDSWEERSTAND)
    with tab_durability:
        st.subheader("🛡️ Durability: Efficiëntie & Drift vs. Verbrande Arbeid (kJ)")
        st.caption("Analyseert hoe goed je lichaam presteert naarmate een rit langer en zwaarder wordt (uitgedrukt in kilojoules arbeid). Een lage decoupling (<5%) en hoge Efficiency Factor bij ritten met veel kJ wijst op een sterke duurmotor.")

        if not df_act.empty and "icu_efficiency_factor" in df_act.columns and "arbeid_kj" in df_act.columns:
            df_dur = df_act[(df_act["icu_efficiency_factor"].notna()) & (df_act["arbeid_kj"] > 50)].copy()
            if not df_dur.empty:
                df_dur["decoupling_plot"] = df_dur["decoupling"].fillna(0.0) if "decoupling" in df_dur.columns else 0.0
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    fig_dur1 = px.scatter(
                        df_dur, x="arbeid_kj", y="icu_efficiency_factor", size="duur_min", color="decoupling_plot",
                        hover_name="name", hover_data=["datum", "weighted_average_watts", "average_heartrate"],
                        color_continuous_scale="RdYlGn_r",
                        title="Efficiency Factor (NP/HR) uitgezet tegen Geleverde Arbeid (kJ)",
                        labels={"arbeid_kj": "Geleverde Arbeid (kJ)", "icu_efficiency_factor": "Efficiency Factor (Watt/HR)", "decoupling_plot": "Drift (%)"},
                    )
                    fig_dur1.update_layout(height=420, template="plotly_dark")
                    st.plotly_chart(fig_dur1)

                with col_d2:
                    if "decoupling" in df_dur.columns and df_dur["decoupling"].notna().any():
                        fig_dur2 = px.scatter(
                            df_dur[df_dur["decoupling"].notna()], x="arbeid_kj", y="decoupling", size="icu_training_load", color="icu_efficiency_factor",
                            hover_name="name", hover_data=["datum", "duur_uur"],
                            color_continuous_scale="Viridis",
                            title="Hartslagdrift (Decoupling %) bij Toenemende Arbeid (kJ)",
                            labels={"arbeid_kj": "Geleverde Arbeid (kJ)", "decoupling": "Aerobe Ontkoppeling / Drift (%)"},
                        )
                        fig_dur2.add_hline(y=5.0, line_dash="dash", line_color="#ef476f", annotation_text="5% Drift Grens (Durability Limiet)")
                        fig_dur2.update_layout(height=420, template="plotly_dark")
                        st.plotly_chart(fig_dur2)
                    else:
                        st.info("Ritten met minimaal 2 laps/intervallen tonen hier automatisch hun hartslagdrift (decoupling %).")
            else:
                st.info("Nog niet genoeg ritten met zowel vermogens- als hartslagdata voor een Durability-analyse.")
        else:
            st.info("Efficiency Factor of kJ data ontbreekt.")

    # TAB 5: VOEDING & MACRONUTRIËNTEN CALCULATOR
    with tab_voeding:
        st.subheader("🍝 Dynamische Energie-, Macro- & In-Ride Fueling Calculator")
        st.caption(f"Berekend op basis van je gewicht (**{actueel_gewicht} kg**), geschat rustmetabolisme en de exacte mechanische arbeid (kJ) van je trainingen per dag.")

        rust_kcal = int((22 * actueel_gewicht) * 1.35)

        if not df_act.empty:
            agg_dict = {
                "ritten": ("name", lambda x: ", ".join(x.dropna())),
                "duur_uur": ("duur_uur", "sum"),
                "arbeid_kj": ("arbeid_kj", "sum"),
            }
            if "icu_intensity" in df_act.columns:
                agg_dict["gem_if"] = ("icu_intensity", "mean")
            dag_energie = df_act.groupby("datum").agg(**agg_dict).reset_index()

            dag_energie["sport_kcal"] = (dag_energie["arbeid_kj"] * 1.05).round(0)
            dag_energie["totaal_kcal"] = rust_kcal + dag_energie["sport_kcal"]

            dag_energie["khd_gram"] = (actueel_gewicht * np.where(dag_energie["duur_uur"] > 2.5, 8.0, np.where(dag_energie["duur_uur"] > 1.2, 6.0, 4.5))).round(0)
            dag_energie["eiwit_gram"] = round(actueel_gewicht * 2.0, 0)
            dag_energie["vet_gram"] = np.maximum(50, ((dag_energie["totaal_kcal"] - (dag_energie["khd_gram"] * 4) - (dag_energie["eiwit_gram"] * 4)) / 9)).round(0)

            dag_energie["in_ride_khd_per_uur"] = np.where(
                dag_energie["duur_uur"] < 1.0, 0,
                np.where(dag_energie["duur_uur"] < 2.0, 45,
                np.where(dag_energie["duur_uur"] < 3.0, 65, 85)),
            )

            fig_nut = go.Figure()
            fig_nut.add_trace(go.Bar(x=dag_energie["datum"], y=dag_energie["khd_gram"] * 4, name="Koolhydraten (kcal)", marker_color="#00b4d8"))
            fig_nut.add_trace(go.Bar(x=dag_energie["datum"], y=dag_energie["eiwit_gram"] * 4, name="Eiwitten (kcal)", marker_color="#06d6a0"))
            fig_nut.add_trace(go.Bar(x=dag_energie["datum"], y=dag_energie["vet_gram"] * 9, name="Vetten (kcal)", marker_color="#ffd166"))
            fig_nut.update_layout(barmode="stack", title="Dagelijkse Calorie- & Macrobehoefte op Trainingsdagen", height=380, template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_nut)

            st.dataframe(
                dag_energie[["datum", "ritten", "duur_uur", "arbeid_kj", "totaal_kcal", "khd_gram", "eiwit_gram", "vet_gram", "in_ride_khd_per_uur"]].sort_values("datum", ascending=False),
                column_config={
                    "datum": "Datum", "ritten": "Sessie(s)", "duur_uur": "Duur (u)",
                    "arbeid_kj": "Arbeid (kJ)", "totaal_kcal": "Totaal Doel (kcal)",
                    "khd_gram": "Koolhydraten (g/dag)", "eiwit_gram": "Eiwit (g/dag)", "vet_gram": "Vetten (g/dag)",
                    "in_ride_khd_per_uur": "Bidon/Gel Advies (g KHD/uur tijdens rit)",
                },
            )

    # TAB 6: GARMIN FYSIOLOGIE, AI DREMPELDETECTIE & RACE PREDICTOR
    with tab_predict:
        st.subheader("🔮 Garmin Fysiologie, Drempeldetectie & Race Predictor")

        if garmin_phys:
            st.markdown("#### ⌚ Officiële Garmin Connect Fysiologie & Biometrie")
            gp1, gp2, gp3, gp4, gp5, gp6 = st.columns(6)
            gp1.metric("VO2max (Fiets)", f"{garmin_phys.get('vo2max_cycling') or '—'}", "ml/kg/min", border=True)
            gp2.metric("VO2max (Lopen)", f"{garmin_phys.get('vo2max_running') or '—'}", "ml/kg/min", border=True)
            gp3.metric("Garmin FTP", f"{garmin_phys.get('cycling_ftp_w') or actueel_ftp} W", f"{round((garmin_phys.get('cycling_ftp_w') or actueel_ftp)/actueel_gewicht, 2)} W/kg", border=True)
            gp4.metric("Omslagpunt (LTHR)", f"{garmin_phys.get('lthr_bpm') or '—'} bpm", f"LT Speed: {garmin_phys.get('lt_speed_kmh') or '—'} km/u", border=True)
            gp5.metric("Endurance & Hill", f"{garmin_phys.get('endurance_score') or '—'}", f"Hill score: {garmin_phys.get('hill_score') or '—'}", border=True)
            acwr_str = f"ACWR {garmin_phys['acwr_ratio']}" if garmin_phys.get("acwr_ratio") else "Garmin Status"
            gp6.metric("Training Status", str(garmin_phys.get("training_status") or "Actief").replace("_", " "), acwr_str, border=True)

        geschat_ftp = actueel_ftp
        if not df_act.empty and "weighted_average_watts" in df_act.columns:
            lange_ritten = df_act[df_act["duur_min"] >= 45]["weighted_average_watts"].dropna()
            if not lange_ritten.empty:
                max_np_45m = lange_ritten.max()
                geschat_ftp = int(max(actueel_ftp, max_np_45m * 0.96))

        lt1_watt = int(geschat_ftp * 0.76)
        sweetspot_range = f"{int(geschat_ftp*0.88)} - {int(geschat_ftp*0.94)} W"
        vo2max_watt = int(geschat_ftp * 1.18)

        st.markdown("#### ⚡ Berekende Trainingsdrempels")
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        c_p1.metric("AI Geschatte eFTP (LT2)", f"{geschat_ftp} W", f"{round(geschat_ftp/actueel_gewicht, 2)} W/kg")
        c_p2.metric("Aerobe Drempel (LT1 / VT1)", f"{lt1_watt} W", "Bovenkant Zone 2 (FatMax)")
        c_p3.metric("Sweetspot Zone", sweetspot_range, "Optimale tijdrit-opbouw")
        c_p4.metric("VO2max Piek (5 min)", f"~{vo2max_watt} W", f"{round(vo2max_watt/actueel_gewicht, 2)} W/kg")

        st.markdown("#### 🏁 Voorspelde Prestaties & Eindtijden (Op basis van actuele eFTP & CTL)")

        def schat_snelheid_kmh(watt, cda=0.25):
            return round(((watt / (0.5 * 1.225 * cda)) ** (1 / 3)) * 3.6 * 0.94, 1)

        tt_20km_watt = int(geschat_ftp * 1.02)
        tt_40km_watt = int(geschat_ftp * 0.96)
        gran_fondo_watt = int(geschat_ftp * 0.78)

        v_20k = schat_snelheid_kmh(tt_20km_watt, cda=0.23)
        v_40k = schat_snelheid_kmh(tt_40km_watt, cda=0.23)
        v_gf = schat_snelheid_kmh(gran_fondo_watt, cda=0.31)

        df_pred = pd.DataFrame([
            {"Onderdeel": "20 km Tijdrit (Aero / TT)", "Doelvermogen": f"{tt_20km_watt} W", "W/kg": round(tt_20km_watt / actueel_gewicht, 2), "Geschatte Snelheid": f"{v_20k} km/u", "Voorspelde Eindtijd": f"{int((20/v_20k)*60)} min {int((((20/v_20k)*60)%1)*60)} sec"},
            {"Onderdeel": "40 km Tijdrit (Aero / TT)", "Doelvermogen": f"{tt_40km_watt} W", "W/kg": round(tt_40km_watt / actueel_gewicht, 2), "Geschatte Snelheid": f"{v_40k} km/u", "Voorspelde Eindtijd": f"{int((40/v_40k)*60)} min {int((((40/v_40k)*60)%1)*60)} sec"},
            {"Onderdeel": "90 km Triathlon Bike Split (Half/70.3)", "Doelvermogen": f"{int(geschat_ftp*0.82)} W", "W/kg": round((geschat_ftp * 0.82) / actueel_gewicht, 2), "Geschatte Snelheid": f"{schat_snelheid_kmh(geschat_ftp*0.82, 0.24)} km/u", "Voorspelde Eindtijd": f"{round(90/schat_snelheid_kmh(geschat_ftp*0.82, 0.24), 2)} uur"},
            {"Onderdeel": "150 km Duurrit / Toertocht (Wegfiets)", "Doelvermogen": f"{gran_fondo_watt} W (NP)", "W/kg": round(gran_fondo_watt / actueel_gewicht, 2), "Geschatte Snelheid": f"{v_gf} km/u", "Voorspelde Eindtijd": f"{round(150/v_gf, 1)} uur"},
        ])
        st.table(df_pred)

    # TAB 7: AI WORKOUT CHAT-STUDIO & GARMIN EXPORTER
    with tab_agenda:
        # Initialiseer session_state voor de Workout Chat & Builder
        standaard_tekst = (
            "Warmup\n- 15m 55% 90rpm\n\nMain Set 3x\n- 8m 90% 95rpm\n- 4m 55% 85rpm\n\nCooldown\n- 10m 50% 85rpm"
            if readiness_score >= 60
            else "Warmup\n- 10m 50% 85rpm\n\nMain Set\n- 45m 65% 90rpm\n\nCooldown\n- 10m 50% 85rpm"
        )
        if "builder_naam" not in st.session_state:
            st.session_state.builder_naam = f"AI Endurance {'Sweetspot 3x8m' if readiness_score >= 60 else 'Herstel / Duur Z2'}"
        if "builder_tekst" not in st.session_state:
            st.session_state.builder_tekst = standaard_tekst
        if "builder_sport" not in st.session_state:
            st.session_state.builder_sport = "Ride"
        if "workout_chat_history" not in st.session_state:
            st.session_state.workout_chat_history = [
                {
                    "role": "assistant",
                    "content": (
                        "Omschrijf hieronder in gewone taal wat voor training je wilt doen "
                        "(bijv. *'Ik wil 75 min fietsen met 3x8 min op sweetspot en 4 min rust'* of "
                        "*'Ik wil een duurloop van 50 min met 5x 1 min op 4:15/km'*). "
                        "Je kunt de gegenereerde workout rechts direct controleren, eventueel bijsturen via de chat "
                        "(bijv. *'maak er 4 herhalingen van'*) en zodra hij oké is met 1 klik exporteren naar Garmin!"
                    ),
                }
            ]

        col_ag1, col_ag2 = st.columns([1.15, 1.35])

        with col_ag1:
            col_wc_hdr, col_wc_clr = st.columns([3.5, 1.2], vertical_alignment="bottom")
            with col_wc_hdr:
                st.subheader("💬 Stap 1: Omschrijf je Workout in de Chat")
            with col_wc_clr:
                if len(st.session_state.workout_chat_history) > 1:
                    if st.button("Reset chat", icon=":material/restart_alt:", width="stretch", key="reset_wc_chat"):
                        st.session_state.pop("workout_chat_history", None)
                        st.rerun()

            st.caption(
                f"De AI houdt automatisch rekening met je **FTP ({actueel_ftp}W)** en je **Daily Readiness ({readiness_score}/100)**."
            )

            # Snelle voorbeelden om met 1 klik te genereren
            ex1, ex2, ex3 = st.columns(3)
            voorbeeld_prompt = None
            if ex1.button("✨ 75m Sweetspot 3x8m", width="stretch"):
                voorbeeld_prompt = "Maak een fietsworkout van 75 minuten met 15 min warming-up, daarna 3x 8 minuten sweetspot (90% FTP, 95 rpm) met 4 minuten herstel ertussen, en 10 min cooling-down."
            if ex2.button("✨ 65m VO2max 5x4m", width="stretch"):
                voorbeeld_prompt = "Maak een pittige VO2max fietsworkout van ongeveer 65 minuten met 5 blokjes van 4 minuten op 112% FTP en 4 minuten rust ertussen."
            if ex3.button("✨ 90m Zone 2 Duurrit", width="stretch"):
                voorbeeld_prompt = "Maak een gevarieerde Zone 2 duurrit van 90 minuten tussen 65% en 73% FTP met af en toe een korte cadansprikkel van 100 rpm."

            # Toon chat-geschiedenis van de Workout Studio in een scrollbare container
            with st.container(height=310, border=True):
                for wc_msg in st.session_state.workout_chat_history:
                    with st.chat_message(wc_msg["role"]):
                        st.markdown(wc_msg["content"])

            with st.form("workout_chat_form", clear_on_submit=True):
                omschrijving_input = st.text_input(
                    "Omschrijf je workout of geef een aanpassing door:",
                    placeholder="Bijv.: '1 uur fietsen met 4x6 min op 270W en 3 min rust' of 'Maak de rust 4 minuten'...",
                    label_visibility="collapsed",
                )
                submit_wc = st.form_submit_button(
                    "✨ Genereer / Pas Workout Aan",
                    icon=":material/auto_awesome:",
                    type="primary",
                    width="stretch",
                )

            actieve_wc_vraag = voorbeeld_prompt if voorbeeld_prompt else (omschrijving_input.strip() if submit_wc and omschrijving_input.strip() else None)

            if actieve_wc_vraag:
                if not gemini_key:
                    st.error("Vul eerst links in de zijbalk je Gemini API Key in.")
                else:
                    st.session_state.workout_chat_history.append({"role": "user", "content": actieve_wc_vraag})
                    wc_hist_str = "\n".join(
                        [f"{m['role']}: {m['content']}" for m in st.session_state.workout_chat_history[-6:]]
                    )
                    wc_prompt = f"""
                    Je bent een gespecialiseerde Garmin Workout Architect voor wielrennen, hardlopen en zwemmen.
                    De atleet omschrijft in gewone taal welke workout hij/zij wil doen of hoe de huidige workout aangepast moet worden.
                    
                    Gegevens atleet:
                    - FTP: {actueel_ftp} Watt | Gewicht: {actueel_gewicht} kg
                    - Daily Readiness vandaag: {readiness_score}/100
                    - Huidige workout in de builder (pas deze aan als de atleet om een wijziging vraagt):
                      Naam: {st.session_state.builder_naam}
                      Sport: {st.session_state.builder_sport}
                      Structuur:
                      {st.session_state.builder_tekst}

                    Gespreksverloop:
                    {wc_hist_str}

                    Nieuwe opdracht van de atleet: "{actieve_wc_vraag}"

                    INSTRUCTIES VOOR JE ANTWOORD:
                    1. Geef een korte, heldere bevestiging in het Nederlands (max 2-3 zinnen) van wat je hebt klaargezet.
                    2. Geef ALTIJD een ```text codeblok met bovenaan 'Naam: ...' en 'Sport: Ride' (of 'Run' / 'Swim'), gevolgd door de exacte stappen:
                    ```text
                    Naam: <Korte pakkende naam van de workout>
                    Sport: <Ride of Run of Swim>
                    Warmup
                    - 15m 55% 90rpm

                    Main Set 3x
                    - 8m 90% 95rpm
                    - 4m 55% 85rpm

                    Cooldown
                    - 10m 50% 85rpm
                    ```
                    Regels voor stappen:
                    - Elke stap begint met '- ' gevolgd door duur ('15m', '45s', '1h') en het doel:
                      - Voor fietsen: percentage FTP ('65%' of '88-93%') of exacte Watts ('250W' of '240-260W') en optioneel cadans ('90rpm').
                      - Voor hardlopen: tempo ('4:30/km' of '4:20-4:35/km') of hartslag ('145bpm' of '140-150bpm').
                    - Gebruik voor herhalingen 'Main Set Nx' (bijv. 'Main Set 4x') gevolgd door de stappen die herhaald moeten worden, en sluit het herhalingsblok af met een lege regel vóór 'Cooldown'.
                    """
                    with st.spinner("Workout wordt opgebouwd..."):
                        ok_wc, antw_wc, _ = vraag_gemini_ai(gemini_key, ai_model, wc_prompt)
                        if ok_wc:
                            gevonden_naam, gevonden_tekst, gevonden_sport = extract_workout_from_ai_response(antw_wc)
                            if gevonden_tekst:
                                st.session_state.builder_tekst = gevonden_tekst
                                if gevonden_naam:
                                    st.session_state.builder_naam = gevonden_naam
                                if gevonden_sport in ("Ride", "Run", "Swim"):
                                    st.session_state.builder_sport = gevonden_sport
                            st.session_state.workout_chat_history.append({"role": "assistant", "content": antw_wc})
                            st.rerun()
                        else:
                            st.error(f"Kon workout niet genereren: {antw_wc}")

        with col_ag2:
            st.subheader("✅ Stap 2: Controleer & Exporteer naar Garmin")
            st.caption("Controleer hieronder de gegenereerde blokken en doelwattages. Is hij helemaal naar wens? Klik dan op **Workout is OK → Exporteer & Plan in Garmin Connect**.")

            with st.container(border=True):
                c_f1, c_f2, c_f3 = st.columns([2.1, 1.15, 0.95])
                with c_f1:
                    w_naam = st.text_input("Workout Naam", key="builder_naam")
                with c_f2:
                    w_datum = st.date_input("Inplannen op datum", datetime.today())
                with c_f3:
                    w_type = st.selectbox("Sport", ["Ride", "Run", "Swim"], key="builder_sport")

                geparseerd = parse_workout_structure(st.session_state.builder_tekst, actueel_ftp)
                if geparseerd["preview_rows"]:
                    m_p1, m_p2, m_p3 = st.columns(3)
                    m_p1.metric("Totale Duur", f"{geparseerd['total_minutes']} min", border=True)
                    m_p2.metric("Geschatte Belasting", f"~{geparseerd['estimated_tss']} TSS", border=True)
                    m_p3.metric("Geschatte Arbeid", f"~{geparseerd['estimated_kj']} kJ", border=True)

                    st.dataframe(pd.DataFrame(geparseerd["preview_rows"]), hide_index=True)
                else:
                    st.warning("Geen geldige workout-stappen herkend. Omschrijf je workout links in de chat!")

                with st.expander("✏️ Handmatig workout-stappen bekijken of aanpassen (optioneel)"):
                    w_beschrijving = st.text_area(
                        "Workout Structuur",
                        key="builder_tekst",
                        height=145,
                    )

                btn_g1, btn_g2 = st.columns([1.4, 1.0])
                with btn_g1:
                    if st.button(
                        "✅ Workout is OK → Exporteer naar Garmin Connect",
                        icon=":material/watch:",
                        type="primary",
                        width="stretch",
                    ):
                        with st.spinner("Workout bouwen, uploaden en inplannen in Garmin Connect..."):
                            ok, res_info = stuur_workout_naar_garmin(
                                naam=w_naam,
                                sport=w_type,
                                datum_str=w_datum.strftime("%Y-%m-%d"),
                                workout_tekst=st.session_state.builder_tekst,
                                ftp=actueel_ftp,
                            )
                            if ok and isinstance(res_info, dict):
                                st.success(
                                    f"✅ **Geëxporteerd & ingepland in Garmin Connect!**\n"
                                    f"- Workout: **{res_info.get('workout_name', w_naam)}** (Garmin ID `{res_info.get('workout_id', '-')}`)\n"
                                    f"- Datum op je Garmin Kalender: **{res_info.get('scheduled_date', w_datum)}**\n"
                                    f"- Open Garmin Connect of synchroniseer je Garmin Edge / horloge en hij staat direct klaar!"
                                )
                                st.toast("Workout geëxporteerd naar Garmin Connect!", icon=":material/check_circle:")
                            else:
                                st.error(f"❌ Garmin export mislukt: {res_info}")

                with btn_g2:
                    if st.button(
                        "☁️ Ook naar Intervals.icu",
                        icon=":material/cloud_upload:",
                        width="stretch",
                        disabled=not heeft_intervals,
                    ):
                        post_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/events"
                        payload = {
                            "category": "WORKOUT",
                            "start_date_local": f"{w_datum.strftime('%Y-%m-%d')}T09:00:00",
                            "name": w_naam,
                            "description": st.session_state.builder_tekst,
                            "type": w_type,
                        }
                        res = requests.post(post_url, auth=("API_KEY", intervals_key), json=payload)
                        if res.status_code in [200, 201]:
                            st.success(f"✅ Workout '{w_naam}' ingepland in Intervals.icu op {w_datum}!")
                            st.cache_data.clear()
                        else:
                            st.error(f"❌ Kon workout niet aanmaken in Intervals.icu: {res.status_code} - {res.text}")

            st.markdown("#### 📅 Gepland in je Garmin Connect Kalender (Komende 14 dagen)")
            if garmin_scheduled:
                df_g_sched = pd.DataFrame(garmin_scheduled)
                toon_sched_cols = [c for c in ["date", "title", "sport", "duration_min", "workout_id"] if c in df_g_sched.columns]
                st.dataframe(
                    df_g_sched[toon_sched_cols],
                    column_config={
                        "date": "Datum",
                        "title": "Workout Naam",
                        "sport": "Sport",
                        "duration_min": "Duur (min)",
                        "workout_id": "Garmin ID",
                    },
                    hide_index=True,
                )
            else:
                st.caption("Nog geen geplande workouts gevonden voor de komende 14 dagen.")

    # TAB 8: RUWE DATA
    with tab_data:
        toon_alles = st.checkbox("🔍 Toon alle ruwe kolommen uit Garmin & Intervals.icu")
        kern_act = ["datum", "name", "type", "duur_min", "afstand_km", "arbeid_kj", "average_watts", "weighted_average_watts", "average_heartrate", "icu_training_load", "icu_intensity", "icu_efficiency_factor", "decoupling"]
        beschikbaar_act = [c for c in kern_act if c in df_act.columns] if not df_act.empty else []

        kern_wel = ["datum", "restingHR", "hrv", "slaap_uren", "sleepScore", "training_readiness", "body_battery_low", "body_battery_high", "stress_avg", "steps", "ctl", "atl", "tsb", "weight"]
        beschikbaar_wel = [c for c in kern_wel if c in df_wel.columns] if not df_wel.empty else []

        st.markdown("**Trainingen**")
        if not df_act.empty:
            drop_complex = [c for c in ["laps", "power_zones", "hr_zones", "icu_zone_times"] if c in df_act.columns]
            st.dataframe(df_act.drop(columns=drop_complex) if toon_alles else df_act[beschikbaar_act])
        st.markdown("**Wellness & Herstel**")
        if not df_wel.empty:
            st.dataframe(df_wel if toon_alles else df_wel[beschikbaar_wel])

    # --- 3. INTERACTIEVE AI PERFORMANCE COACH & WORKOUT CHAT ---
    st.divider()
    col_coach_hdr, col_coach_clr = st.columns([4, 1], vertical_alignment="bottom")
    with col_coach_hdr:
        st.subheader("🧠 AI Performance Coach & Workout Chat (Omschrijf → Controleer → Exporteer naar Garmin)")
    with col_coach_clr:
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            if st.button("Gesprek wissen", icon=":material/delete:", width="stretch"):
                st.session_state.pop("messages", None)
                st.rerun()

    if not gemini_key:
        if st.session_state.pop("trigger_laatste_workout_analyse", False):
            st.info(
                "Je Garmin workout is gesynchroniseerd! Vul links in de zijbalk eenmalig je **Gemini API Key** in (deze wordt automatisch opgeslagen) om direct je post-workout analyse te starten."
            )
        st.warning("👉 Vul links in de zijbalk eenmalig je **Gemini API Key** in om de AI-analyse en Workout Generator te gebruiken (wordt automatisch bewaard).")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"Je data van de afgelopen {dagen} dagen is ingeladen (**Daily Readiness: {readiness_score}/100** | **FTP: {actueel_ftp}W**). "
                        f"Je kunt hier in de chat **direct je gewenste workout omschrijven** (bijv. *'Maak een fietsworkout van 1,5 uur met 4x8 min sweetspot'* of *'Pas hem aan naar 3x10 min'*). "
                        f"Ik toon dan direct een overzichtstabel van alle blokken zodat je kunt controleren of hij oké is en hem met **1 klik naar Garmin Connect** kunt exporteren!"
                    ),
                }
            ]

        b0, b1, b2, b3, b4 = st.columns(5)
        snelle_vraag = None
        if b0.button("Hersync & Analyseer Laatste Rit", icon=":material/bolt:", type="primary", width="stretch"):
            with st.spinner("Laatste Garmin workout & interval-laps ophalen..."):
                ok, msg = voer_garmin_sync_uit(2, force_details=True)
                if ok:
                    st.session_state.trigger_laatste_workout_analyse = True
                    st.rerun()
                else:
                    st.error(f"Garmin sync mislukt: {msg}")
        if b1.button("Week- & Readiness Analyse", icon=":material/analytics:", width="stretch"):
            snelle_vraag = "Geef een scherpe analyse van mijn afgelopen 7 dagen. Hoe verhouden mijn geleverde wattages, kJ arbeid en trainingsload zich tot mijn Garmin Training Readiness, Body Battery, HRV, slaap en rusthartslag?"
        if b2.button("Durability & Lap-Drift Check", icon=":material/shield:", width="stretch"):
            snelle_vraag = "Analyseer mijn Efficiency Factor, Decoupling (hartslagdrift) en de interval/lap-opbouw van mijn recente ritten. Hoe staat het met mijn vermoeidheidsweerstand (durability)?"
        if b3.button("Genereer Workout voor Vandaag", icon=":material/watch:", width="stretch"):
            snelle_vraag = (
                f"Mijn Readiness Score is vandaag {readiness_score}/100 en mijn FTP is {actueel_ftp}W. "
                f"Stel de optimale workout voor vandaag samen op basis van mijn recente belasting (CTL/ATL/TSB) en herstel, "
                f"zodat ik de blokken kan controleren en direct kan exporteren naar Garmin Connect."
            )
        if b4.button("Voedingsplan voor morgen", icon=":material/restaurant:", width="stretch"):
            snelle_vraag = f"Op basis van mijn recente trainingsbelasting en gewicht ({actueel_gewicht} kg): geef een concreet dagmenu en fueling-strategie (voor, tijdens en na de rit) voor een training van 2,5 uur morgen."

        if st.session_state.pop("trigger_laatste_workout_analyse", False):
            if not df_act.empty:
                laatste = df_act.iloc[-1]
                l_naam = laatste.get("name", "Workout")
                l_datum = laatste.get("datum", "vandaag")
                l_duur = laatste.get("duur_min", "-")
                l_watt = laatste.get("average_watts", "-")
                l_np = laatste.get("weighted_average_watts", "-")
                l_hr = laatste.get("average_heartrate", "-")
                l_load = laatste.get("icu_training_load", "-")
                l_ef = laatste.get("icu_efficiency_factor", "-")
                l_dec = laatste.get("decoupling", "-")
                snelle_vraag = (
                    f"Ik heb net mijn Garmin gesynchroniseerd na mijn laatste workout: '{l_naam}' op {l_datum} "
                    f"(duur: {l_duur} min, gem. vermogen: {l_watt} W, NP: {l_np} W, gem. HR: {l_hr} bpm, load: {l_load} TSS, EF: {l_ef}, decoupling: {l_dec}%). "
                    f"Analyseer deze sessie én de afzonderlijke laps/intervallen ten opzichte van mijn FTP ({actueel_ftp}W) "
                    f"en mijn huidige herstelcijfers (Readiness {readiness_score}/100). Geef ook concreet herstel- en voedingsadvies voor de komende 24 uur "
                    f"én voeg onderaan een voorstel-workout voor morgen toe zodat ik die kan controleren en naar Garmin Connect kan exporteren."
                )
            else:
                snelle_vraag = "Ik heb net mijn Garmin gesynchroniseerd. Analyseer mijn meest recente workout en mijn huidige herstelstatus."

        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Als dit een AI-bericht is waarin een workout-blok staat, toon direct de Visuele Controle-tabel + 1-Klik Garmin Export!
                if msg["role"] == "assistant":
                    gevonden_titel, gevonden_workout, gevonden_sport = extract_workout_from_ai_response(msg["content"])
                    if gevonden_workout:
                        ai_parsed = parse_workout_structure(gevonden_workout, actueel_ftp)
                        if ai_parsed["preview_rows"]:
                            with st.container(border=True):
                                st.markdown(
                                    f"#### 🔍 Controleer Gegenereerde Workout: `{gevonden_titel}`\n"
                                    f"**Totale duur:** {ai_parsed['total_minutes']} min &nbsp;|&nbsp; "
                                    f"**Geschatte Load:** ~{ai_parsed['estimated_tss']} TSS &nbsp;|&nbsp; "
                                    f"**Arbeid:** ~{ai_parsed['estimated_kj']} kJ *(o.b.v. FTP {actueel_ftp}W)*"
                                )
                                st.dataframe(pd.DataFrame(ai_parsed["preview_rows"]), hide_index=True)
                                st.caption("💡 *Wil je iets aanpassen? Typ het gewoon hieronder in de chat (bijv. 'maak de blokjes 6 min' of '10W lager'). Is hij goed? Klik dan op Exporteren:*")

                                sport_opties = ["Ride", "Run", "Swim"]
                                std_sport_idx = sport_opties.index(gevonden_sport) if gevonden_sport in sport_opties else 0

                                col_ai_w1, col_ai_w2, col_ai_w3, col_ai_w4 = st.columns([1.8, 0.85, 1.05, 1.7], vertical_alignment="bottom")
                                with col_ai_w1:
                                    ai_w_naam = st.text_input(
                                        "Workout naam",
                                        value=gevonden_titel or "AI Coach Garmin Workout",
                                        key=f"ai_w_name_{msg_idx}",
                                    )
                                with col_ai_w2:
                                    ai_w_sport = st.selectbox(
                                        "Sport",
                                        sport_opties,
                                        index=std_sport_idx,
                                        key=f"ai_w_sport_{msg_idx}",
                                    )
                                with col_ai_w3:
                                    ai_w_datum = st.date_input(
                                        "Plandatum",
                                        value=datetime.today(),
                                        key=f"ai_w_date_{msg_idx}",
                                    )
                                with col_ai_w4:
                                    if st.button(
                                        "✅ Workout is OK → Exporteer naar Garmin",
                                        icon=":material/watch:",
                                        type="primary",
                                        width="stretch",
                                        key=f"ai_push_garmin_{msg_idx}",
                                    ):
                                        with st.spinner("Direct exporteren & inplannen in Garmin Connect..."):
                                            ok_g, info_g = stuur_workout_naar_garmin(
                                                naam=ai_w_naam,
                                                sport=ai_w_sport,
                                                datum_str=ai_w_datum.strftime("%Y-%m-%d"),
                                                workout_tekst=gevonden_workout,
                                                ftp=actueel_ftp,
                                            )
                                            if ok_g and isinstance(info_g, dict):
                                                st.success(
                                                    f"✅ **'{ai_w_naam}' staat nu in je Garmin Connect kalender op {ai_w_datum.strftime('%Y-%m-%d')}** "
                                                    f"(Garmin ID `{info_g.get('workout_id')}`)!"
                                                )
                                                st.toast("Geëxporteerd naar Garmin Connect!", icon=":material/check_circle:")
                                            else:
                                                st.error(f"Kon niet naar Garmin exporteren: {info_g}")

        chat_input = st.chat_input("Omschrijf de workout die je wilt genereren/aanpassen, of stel een vraag aan je coach...")
        gebruikers_vraag = snelle_vraag if snelle_vraag else chat_input

        if gebruikers_vraag:
            st.session_state.messages.append({"role": "user", "content": gebruikers_vraag})
            with st.chat_message("user"):
                st.markdown(gebruikers_vraag)

            act_data_str = df_act[beschikbaar_act].to_string() if not df_act.empty and beschikbaar_act else "Geen trainingen"
            wel_data_str = df_wel[beschikbaar_wel].to_string() if not df_wel.empty and beschikbaar_wel else "Geen wellness data"

            laatste_laps_str = "Geen lap-splits beschikbaar."
            if not df_act.empty and "laps" in df_act.columns:
                laatste_rij = df_act.iloc[-1]
                if isinstance(laatste_rij.get("laps"), list) and laatste_rij["laps"]:
                    laatste_laps_str = (
                        f"Lap-splits van laatste workout ({laatste_rij.get('name')} op {laatste_rij.get('datum')}):\n"
                        + pd.DataFrame(laatste_rij["laps"]).to_string(index=False)
                    )

            phys_str = json.dumps(garmin_phys, ensure_ascii=False) if garmin_phys else "Geen extra fysiologie"
            gesprek_historie = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])

            prompt = f"""
            Je bent een analytische, no-nonsense performance coach én Garmin Workout Architect voor wielrennen, hardlopen en triatlon.
            Je spreekt duidelijk en direct Nederlands, zonder wollig of typisch AI-taalgebruik.
            Onderbouw elke conclusie direct met harde cijfers uit de tabellen hieronder.
            
            BELANGRIJK BIJ HET GENEREREN OF AANPASSEN VAN EEN WORKOUT:
            Als de atleet in de chat een workout omschrijft (of vraagt om een workout te genereren of aan te passen), zet de workout dan ALTIJD in een ```text codeblok in exact dit formaat:
            ```text
            Naam: <Pakkende naam van de workout>
            Sport: <Ride of Run of Swim>
            Warmup
            - 15m 55% 90rpm

            Main Set 3x
            - 8m 90% 95rpm
            - 4m 55% 85rpm

            Cooldown
            - 10m 50% 85rpm
            ```
            Regels voor de workout-stappen in het codeblok:
            - Elke stap begint met '- ' gevolgd door de duur ('15m', '45s', '1h') en het doel:
              - Voor fietsen (Sport: Ride): percentage FTP ('65%' of '88-93%') of exacte Watts ('250W' of '240-260W') en optioneel cadans ('90rpm').
              - Voor hardlopen (Sport: Run): tempo ('4:30/km' of '4:20-4:35/km') of hartslag ('145bpm' of '140-150bpm').
            - Gebruik voor herhalingsblokken 'Main Set Nx' (bijv. 'Main Set 4x') en laat een lege regel vóór 'Cooldown'.
            Dankzij dit codeblok krijgt de atleet direct een visuele controle-tabel en een 1-klik knop 'Workout is OK -> Exporteer naar Garmin'.

            Actuele parameters atleet:
            - FTP: {actueel_ftp} Watt | Gewicht: {actueel_gewicht} kg ({round(actueel_ftp/actueel_gewicht, 2)} W/kg)
            - Berekende Daily Readiness Score vandaag: {readiness_score}/100 ({readiness_details})
            - Garmin Fysiologie & Drempels: {phys_str}

            === TRAINING DATA AFGELOPEN {dagen} DAGEN ===
            {act_data_str}

            === LAATSTE WORKOUT INTERVAL / LAP SPLITS ===
            {laatste_laps_str}

            === HERSTEL & GARMIN WELLNESS DATA AFGELOPEN {dagen} DAGEN ===
            {wel_data_str}

            === EERDERE BERICHTEN IN DIT GESPREK ===
            {gesprek_historie}

            Vraag / Workout-omschrijving van de atleet: {gebruikers_vraag}
            """

            with st.chat_message("assistant"):
                with st.spinner("Workout / analyse wordt opgebouwd..."):
                    ok, antwoord_of_fout, gebruikt_model = vraag_gemini_ai(gemini_key, ai_model, prompt)
                    if ok:
                        gev_naam, gev_tekst, gev_sport = extract_workout_from_ai_response(antwoord_of_fout)
                        if gev_tekst:
                            st.session_state.builder_tekst = gev_tekst
                            if gev_naam:
                                st.session_state.builder_naam = gev_naam
                            if gev_sport in ("Ride", "Run", "Swim"):
                                st.session_state.builder_sport = gev_sport
                        st.session_state.messages.append({"role": "assistant", "content": antwoord_of_fout})
                        st.rerun()
                    else:
                        st.error(
                            f"Fout bij AI-aanvraag: {antwoord_of_fout}. "
                            "Kies links in het menu een ander model of controleer je Gemini API Key."
                        )

else:
    st.info("👈 Klik links in de zijbalk op **Hersync laatste workout & analyseer** of vul je **Intervals.icu** gegevens in om het dashboard te activeren.")
