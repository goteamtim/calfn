# natural_v1 review sheet

Now: Wed 2026-10-14 09:00. Reviewed by a human: **NO**.

| id | prompt | expected |
|---|---|---|
| natural-000 | dentist tomorrow at 3pm | `{"title": "Dentist", "start": "2026-10-15T15:00"}` |
| natural-001 | Can you put lunch with Sam on my calendar for Friday at noon? | `{"title": "Lunch with Sam", "start": "2026-10-16T12:00"}` |
| natural-002 | team standup every weekday at 9:30am | `{"title": "Team standup", "start": "2026-10-15T09:30", "recurrence_freq": "weekly", "recurrence_days": "MO,TU,WE,TH,FR"}` |
| natural-003 | remind me to call mom on Sunday at 6pm | `{"title": "Call mom", "start": "2026-10-18T18:00"}` |
| natural-004 | book a haircut next Monday 10am | `{"title": "Haircut", "start": "2026-10-19T10:00"}` |
| natural-005 | yoga every Tuesday and Thursday at 6:30pm | `{"title": "Yoga", "start": "2026-10-15T18:30", "recurrence_freq": "weekly", "recurrence_days": "TU,TH"}` |
| natural-006 | dinner with the Hendersons October 24 at 7 | **reject** |
| natural-007 | meeting with Priya on the 20th at 2:30 pm for 45 minutes | `{"title": "Meeting with Priya", "start": "2026-10-20T14:30", "duration_minutes": 45}` |
| natural-008 | pick up the kids at 3:15 today | `{"title": "Pick up the kids", "start": "2026-10-14T15:15"}` |
| natural-009 | Doctor's appointment Thursday Oct 22 at 11am | `{"title": "Doctor's appointment", "start": "2026-10-22T11:00"}` |
| natural-010 | weekly sync Mondays at 10am starting Oct 26 | `{"title": "Weekly sync", "start": "2026-10-26T10:00", "recurrence_freq": "weekly", "recurrence_days": "MO"}` |
| natural-011 | Book club the first Wednesday of next month at 7:30pm | `{"title": "Book club", "start": "2026-11-04T19:30"}` |
| natural-012 | mom's birthday every year on March 3rd at 9am | `{"title": "Mom's birthday", "start": "2027-03-03T09:00", "recurrence_freq": "yearly"}` |
| natural-013 | tell me a joke | **reject** |
| natural-014 | what's on my calendar tomorrow? | **reject** |
| natural-015 | schedule a call with the plumber sometime next week | **reject** |
| natural-016 | gym tomorrow morning | **reject** |
| natural-017 | Thanksgiving dinner Nov 26 at 4pm at Aunt Rosa's | `{"title": "Thanksgiving dinner", "start": "2026-11-26T16:00", "location": "Aunt Rosa's"}` |
| natural-018 | Flight to Denver Friday at 6:45am | `{"title": "Flight to Denver", "start": "2026-10-16T06:45"}` |
| natural-019 | 1:1 with Jordan on Tuesday at 2pm | `{"title": "1:1 with Jordan", "start": "2026-10-20T14:00"}` |
| natural-020 | pay rent on the 1st at 8am every month | `{"title": "Pay rent", "start": "2026-11-01T08:00", "recurrence_freq": "monthly"}` |
| natural-021 | coffee with Maria at 10 tomorrow | **reject** |
| natural-022 | Parent teacher conference Oct 29 from 4pm to 5:30pm in Room 12 | `{"title": "Parent teacher conference", "start": "2026-10-29T16:00", "duration_minutes": 90, "location": "Room 12"}` |
| natural-023 | Dentist Friday | **reject** |
| natural-024 | daily standup at 9:15am for 10 days | `{"title": "Daily standup", "start": "2026-10-15T09:15", "recurrence_freq": "daily", "recurrence_count": 10}` |
| natural-025 | Dinner at 7pm | **reject** |
| natural-026 | walk the dog every other day at 7am | `{"title": "Walk the dog", "start": "2026-10-15T07:00", "recurrence_freq": "daily", "recurrence_interval": 2}` |
| natural-027 | Zoom call with the client Wed Oct 21 at 3pm for an hour | `{"title": "Zoom call with the client", "start": "2026-10-21T15:00", "duration_minutes": 60}` |
| natural-028 | Piano recital December 12th at 2pm at the community center | `{"title": "Piano recital", "start": "2026-12-12T14:00", "location": "the community center"}` |
| natural-029 | water the plants every Sunday evening | **reject** |
| natural-030 | Lunch w/ Dad in 2 weeks at 12:30pm | `{"title": "Lunch w/ Dad", "start": "2026-10-28T12:30"}` |
| natural-031 | Make a reminder for the 5k race on November 7th at 8am | `{"title": "5k race", "start": "2026-11-07T08:00"}` |
