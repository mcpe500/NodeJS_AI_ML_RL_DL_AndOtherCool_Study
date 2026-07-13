#!/usr/bin/env python3
"""Synthesize function-calling JSONL for needle.c (Termux-first sizes)."""
from __future__ import annotations
import argparse
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent
CATALOG = json.loads((ROOT / "tools_catalog.json").read_text(encoding="utf-8"))

LOCATIONS = ["San Francisco", "Jakarta", "Tokyo", "London", "New York", "Bandung", "Paris"]
TIMES = ["06:30", "07:00", "12:00", "18:45", "21:15"]
QUERIES = {
    "get_weather": [
        "What's the weather in {location}?",
        "How's the weather in {location} today?",
        "Cuaca di {location} bagaimana?",
        "weather forecast {location}",
    ],
    "create_alarm": [
        "set an alarm for {time}",
        "wake me up at {time}",
        "alarm jam {time}",
    ],
    "web_search": [
        "search for {q}",
        "google {q}",
        "cari {q}",
    ],
    "set_light": [
        "turn {on} the lights in the {room}",
        "set {room} lights {on}",
    ],
    "send_email": [
        "email {to} about {subject}",
        "send mail to {to}: {subject}",
    ],
    "play_music": [
        "play {song}",
        "put on {song}",
    ],
    "get_time": [
        "what time is it?",
        "jam berapa sekarang?",
    ],
    "translate": [
        "translate '{text}' to {lang}",
        "terjemahkan '{text}' ke {lang}",
    ],
    "set_timer": [
        "set a timer for {minutes} minutes",
        "timer {minutes} menit",
    ],
    "open_app": [
        "open {app}",
        "buka aplikasi {app}",
    ],
}
NO_TOOL = [
    "hello", "hi", "thanks", "terima kasih", "how are you?",
    "good morning", "ok", "bye", "what is 2+2?", "tell me a joke",
]
ROOMS = ["kitchen", "bedroom", "living room", "office"]
SONGS = ["Bohemian Rhapsody", "Shape of You", "Laskar Pelangi"]
APPS = ["maps", "calendar", "messages", "camera"]
LANGS = ["id", "en", "ja", "es"]
TEXTS = ["hello", "good morning", "terima kasih"]
SUBJECTS = ["meeting", "invoice", "hello"]
QS = ["needle papers", "best ramen", "python tutorial", "cuaca hari ini"]


def tool_schema(t):
    return {"name": t["name"], "parameters": t["parameters"]}


def fill_args(name: str, rng: random.Random) -> dict:
    if name == "get_weather":
        return {"location": rng.choice(LOCATIONS)}
    if name == "create_alarm":
        return {"time": rng.choice(TIMES)}
    if name == "web_search":
        return {"q": rng.choice(QS)}
    if name == "set_light":
        return {"room": rng.choice(ROOMS), "on": rng.choice([True, False])}
    if name == "send_email":
        return {"to": f"user{rng.randint(1,9)}@example.com", "subject": rng.choice(SUBJECTS)}
    if name == "play_music":
        return {"song": rng.choice(SONGS)}
    if name == "get_time":
        return {}
    if name == "translate":
        return {"text": rng.choice(TEXTS), "lang": rng.choice(LANGS)}
    if name == "set_timer":
        return {"minutes": rng.choice([5, 10, 15, 30])}
    if name == "open_app":
        return {"app": rng.choice(APPS)}
    return {}


def make_query(name: str, args: dict, rng: random.Random) -> str:
    tmpls = QUERIES[name]
    t = rng.choice(tmpls)
    # map boolean on → "on"/"off" words
    fmt = dict(args)
    if "on" in fmt:
        fmt["on"] = "on" if fmt["on"] else "off"
    try:
        return t.format(**fmt)
    except KeyError:
        return t


def sample_row(rng: random.Random, no_tool_p=0.15) -> dict:
    if rng.random() < no_tool_p:
        tools = [tool_schema(t) for t in rng.sample(CATALOG, k=rng.randint(1, 4))]
        return {"query": rng.choice(NO_TOOL), "tools": tools, "call": []}

    primary = rng.choice(CATALOG)
    args = fill_args(primary["name"], rng)
    query = make_query(primary["name"], args, rng)

    # distractors
    others = [t for t in CATALOG if t["name"] != primary["name"]]
    n_dist = rng.randint(0, min(3, len(others)))
    tools = [tool_schema(primary)] + [tool_schema(t) for t in rng.sample(others, n_dist)]
    rng.shuffle(tools)

    call = [{"name": primary["name"], "arguments": args}]
    return {"query": query, "tools": tools, "call": call}


def write_split(path: pathlib.Path, n: int, seed: int):
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(json.dumps(sample_row(rng), ensure_ascii=False) + "\n")
    print(f"wrote {path} n={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT.parent / "raw"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    write_split(out / "train.jsonl", args.train, 1)
    write_split(out / "val.jsonl", args.val, 2)
    write_split(out / "test.jsonl", args.test, 3)


if __name__ == "__main__":
    main()
