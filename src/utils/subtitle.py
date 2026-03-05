import math
from typing import List
from src.db.models import Segment

def format_timestamp(seconds: float, is_vtt: bool = False) -> str:
    """
    SRT: 00:00:01,500
    VTT: 00:00:01.500
    """
    total_milliseconds = int(round(seconds * 1000))
    millis = total_milliseconds % 1000
    total_seconds = total_milliseconds // 1000
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60

    separator = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"

def generate_subtitle_content(segments: List[Segment], is_translation: bool, is_vtt: bool = False) -> str:
    lines = []
    if is_vtt:
        lines.append("WEBVTT\n\n")

    counter = 1
    for seg in segments:
        text = seg.translation_text if is_translation else seg.transcription_text
        
        if not text or not text.strip():
            continue

        start_time = format_timestamp(seg.start_timestamp, is_vtt)
        end_time = format_timestamp(seg.end_timestamp, is_vtt)

        if not is_vtt:
            lines.append(str(counter))
            
        lines.append(f"{start_time} --> {end_time}")
        lines.append(text.strip())
        lines.append("") # Blank as divider
        
        counter += 1

    return "\n".join(lines)