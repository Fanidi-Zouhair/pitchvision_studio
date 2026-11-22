import math


def compute_match_summary(timeline: list, fps: float = 30.0) -> dict:
    """Kay-7sab l-i7sa2iyat l-asasiya mn JSON timeline (distance, max speed, radar stability)."""
    total = len(timeline)
    ok_frames = sum(1 for f in timeline if f.get("radar_status") == "OK")
    player_dist_m = {}
    last_pos = {}

    for f in timeline:
        for p in f.get("players", []):
            pid = p["id"]
            pos = (p["x"], p["y"])
            if pid in last_pos:
                d = math.hypot(pos[0] - last_pos[pid][0], pos[1] - last_pos[pid][1])
                if d < 1.5:  # Ignore teleport jumps (>1.5m/frame)
                    player_dist_m[pid] = player_dist_m.get(pid, 0.0) + d
            last_pos[pid] = pos

    return {
        "total_frames": total,
        "radar_ok_pct": round((ok_frames / max(total, 1)) * 100.0, 1),
        "tracked_players": len(player_dist_m),
        "distances_m": {str(k): round(v, 1) for k, v in player_dist_m.items()},
    }
