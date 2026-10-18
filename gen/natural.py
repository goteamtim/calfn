"""
Hand-drafted "natural" prompts: plain wording that no template produced.

    python -m gen.natural        # writes data/eval/natural_v1.jsonl and natural_v1_review.md

All rows share one "now": Wednesday 2026-10-14 09:00. Labels here are written by hand (by Claude,
in the session that built the project), NOT computed, so unlike eval_v1 / test_v1 they need a human
review before their score means anything. Report this slice separately and don't fold it into the
headline number until it has been reviewed (set REVIEWED = True after you have checked every row).

Calendar for reference: Thu 10/15, Fri 10/16, Sat 10/17, Sun 10/18, Mon 10/19, Tue 10/20, Wed 10/21,
Thu 10/22, Fri 10/23, Mon 10/26, Thu 10/29, Sun 11/1, Sat 11/7, Wed 11/4, Thu 11/26, Sat 12/12.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calfn.schema import direct_developer_message, parse_start, validate_event  # noqa: E402

REVIEWED = False
NOW = datetime(2026, 10, 14, 9, 0)

REJECT = None

ROWS = [
    ("dentist tomorrow at 3pm",
     {"title": "Dentist", "start": "2026-10-15T15:00"}),
    ("Can you put lunch with Sam on my calendar for Friday at noon?",
     {"title": "Lunch with Sam", "start": "2026-10-16T12:00"}),
    ("team standup every weekday at 9:30am",
     {"title": "Team standup", "start": "2026-10-15T09:30", "recurrence_freq": "weekly",
      "recurrence_days": "MO,TU,WE,TH,FR"}),
    ("remind me to call mom on Sunday at 6pm",
     {"title": "Call mom", "start": "2026-10-18T18:00"}),
    ("book a haircut next Monday 10am",
     {"title": "Haircut", "start": "2026-10-19T10:00"}),
    ("yoga every Tuesday and Thursday at 6:30pm",
     {"title": "Yoga", "start": "2026-10-15T18:30", "recurrence_freq": "weekly", "recurrence_days": "TU,TH"}),
    ("dinner with the Hendersons October 24 at 7", REJECT),            # bare hour, no am/pm
    ("meeting with Priya on the 20th at 2:30 pm for 45 minutes",
     {"title": "Meeting with Priya", "start": "2026-10-20T14:30", "duration_minutes": 45}),
    ("pick up the kids at 3:15 today",
     {"title": "Pick up the kids", "start": "2026-10-14T15:15"}),
    ("Doctor's appointment Thursday Oct 22 at 11am",
     {"title": "Doctor's appointment", "start": "2026-10-22T11:00"}),
    ("weekly sync Mondays at 10am starting Oct 26",
     {"title": "Weekly sync", "start": "2026-10-26T10:00", "recurrence_freq": "weekly", "recurrence_days": "MO"}),
    ("Book club the first Wednesday of next month at 7:30pm",
     {"title": "Book club", "start": "2026-11-04T19:30"}),
    ("mom's birthday every year on March 3rd at 9am",
     {"title": "Mom's birthday", "start": "2027-03-03T09:00", "recurrence_freq": "yearly"}),
    ("tell me a joke", REJECT),
    ("what's on my calendar tomorrow?", REJECT),
    ("schedule a call with the plumber sometime next week", REJECT),   # vague
    ("gym tomorrow morning", REJECT),                                  # vague time
    ("Thanksgiving dinner Nov 26 at 4pm at Aunt Rosa's",
     {"title": "Thanksgiving dinner", "start": "2026-11-26T16:00", "location": "Aunt Rosa's"}),
    ("Flight to Denver Friday at 6:45am",
     {"title": "Flight to Denver", "start": "2026-10-16T06:45"}),
    ("1:1 with Jordan on Tuesday at 2pm",
     {"title": "1:1 with Jordan", "start": "2026-10-20T14:00"}),
    ("pay rent on the 1st at 8am every month",
     {"title": "Pay rent", "start": "2026-11-01T08:00", "recurrence_freq": "monthly"}),
    ("coffee with Maria at 10 tomorrow", REJECT),                      # bare hour, no am/pm
    ("Parent teacher conference Oct 29 from 4pm to 5:30pm in Room 12",
     {"title": "Parent teacher conference", "start": "2026-10-29T16:00", "duration_minutes": 90,
      "location": "Room 12"}),
    ("Dentist Friday", REJECT),                                        # no time
    ("daily standup at 9:15am for 10 days",
     {"title": "Daily standup", "start": "2026-10-15T09:15", "recurrence_freq": "daily", "recurrence_count": 10}),
    ("Dinner at 7pm", REJECT),                                         # no date
    ("walk the dog every other day at 7am",
     {"title": "Walk the dog", "start": "2026-10-15T07:00", "recurrence_freq": "daily", "recurrence_interval": 2}),
    ("Zoom call with the client Wed Oct 21 at 3pm for an hour",
     {"title": "Zoom call with the client", "start": "2026-10-21T15:00", "duration_minutes": 60}),
    ("Piano recital December 12th at 2pm at the community center",
     {"title": "Piano recital", "start": "2026-12-12T14:00", "location": "the community center"}),
    ("water the plants every Sunday evening", REJECT),                 # vague time
    ("Lunch w/ Dad in 2 weeks at 12:30pm",
     {"title": "Lunch w/ Dad", "start": "2026-10-28T12:30"}),
    ("Make a reminder for the 5k race on November 7th at 8am",
     {"title": "5k race", "start": "2026-11-07T08:00"}),
]

REJECTION_TEXT = "I can only create an event when I have a specific date and time."


def build() -> list[dict]:
    rows = []
    for i, (text, gold) in enumerate(ROWS):
        if gold is not None:
            ok, msg = validate_event(gold)
            assert ok, (text, msg)
            assert parse_start(gold["start"]) > NOW, text
            asst = {"role": "assistant", "content": None, "tool_calls": [{
                "type": "function", "function": {"name": "create_event", "arguments": gold}}]}
        else:
            asst = {"role": "assistant", "content": REJECTION_TEXT}
        rows.append({
            "messages": [{"role": "developer", "content": direct_developer_message(NOW)},
                         {"role": "user", "content": text}, asst],
            "meta": {"id": f"natural-{i:03d}", "now": NOW.strftime("%Y-%m-%dT%H:%M"),
                     "tags": ["natural"] + (["reject"] if gold is None else []), "reviewed": REVIEWED},
        })
    return rows


def main() -> None:
    out = Path("data/eval")
    rows = build()
    with open(out / "natural_v1.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = ["# natural_v1 review sheet", "",
             f"Now: Wed 2026-10-14 09:00. Reviewed by a human: **{'yes' if REVIEWED else 'NO'}**.", "",
             "| id | prompt | expected |", "|---|---|---|"]
    for r in rows:
        call = r["messages"][2].get("tool_calls")
        expected = f"`{json.dumps(call[0]['function']['arguments'], ensure_ascii=False)}`" if call else "**reject**"
        lines.append(f"| {r['meta']['id']} | {r['messages'][1]['content']} | {expected} |")
    (out / "natural_v1_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows ({sum(1 for r in rows if 'reject' in r['meta']['tags'])} rejections)")


if __name__ == "__main__":
    main()
