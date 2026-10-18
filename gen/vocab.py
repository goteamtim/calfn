"""
Vocabulary for the generator, split into TRAIN and HELDOUT halves.

Eval and test draw mostly from HELDOUT (sentence frames, event titles, locations and
off-topic prompts the model never saw in training) so a good score means the model
generalized, not that it memorized templates. A smaller share uses TRAIN vocabulary with
fresh dates and combinations, which lets results report an in-distribution vs held-out gap.

Titles are written the way a person says them (lowercase, "dentist appointment"). The label
is the same text with its first letter capitalized.

Frame kinds: "any" fits every title, "verb" needs a verb phrase (call mom), "noun" needs a
noun phrase (dentist appointment).
"""

# ------------------------------------------------------------------ titles: (text, kind)

TRAIN_TITLES = [
    # noun phrases
    ("dentist appointment", "noun"), ("lunch with Sam", "noun"), ("team standup", "noun"),
    ("yoga class", "noun"), ("haircut", "noun"), ("piano lesson", "noun"),
    ("dinner with the Hendersons", "noun"), ("doctor's appointment", "noun"),
    ("book club", "noun"), ("date night", "noun"), ("weekly sync", "noun"),
    ("coffee with Maria", "noun"), ("parent teacher conference", "noun"),
    ("physical therapy", "noun"), ("soccer practice", "noun"), ("budget review", "noun"),
    ("project kickoff", "noun"), ("dog grooming", "noun"), ("oil change", "noun"),
    ("job interview", "noun"), ("therapy session", "noun"), ("spin class", "noun"),
    ("choir rehearsal", "noun"), ("game night", "noun"), ("design review", "noun"),
    ("lunch with Dad", "noun"), ("breakfast with Priya", "noun"), ("vet appointment", "noun"),
    ("movie night", "noun"), ("team retro", "noun"), ("language exchange", "noun"),
    ("eye exam", "noun"), ("car inspection", "noun"), ("sprint planning", "noun"),
    ("happy hour with the team", "noun"), ("guitar lesson", "noun"), ("swim practice", "noun"),
    ("farmers market", "noun"), ("pottery class", "noun"), ("book signing", "noun"),
    ("client call", "noun"), ("all hands", "noun"), ("quarterly planning", "noun"),
    ("neighborhood meeting", "noun"), ("art class", "noun"), ("tax appointment", "noun"),
    ("dinner for 4", "noun"), ("study session for chapter 5", "noun"),
    ("1:1 with Jordan", "noun"), ("flight check-in", "noun"), ("birthday party", "noun"),
    # verb phrases
    ("call mom", "verb"), ("pick up the kids", "verb"), ("go to the gym", "verb"),
    ("water the plants", "verb"), ("submit the report", "verb"), ("renew my passport", "verb"),
    ("take out the trash", "verb"), ("pay rent", "verb"), ("email the landlord", "verb"),
    ("walk the dog", "verb"), ("meal prep", "verb"), ("back up my laptop", "verb"),
    ("call the plumber", "verb"), ("buy groceries", "verb"), ("clean the garage", "verb"),
    ("call grandma", "verb"), ("file my expenses", "verb"), ("pick up the dry cleaning", "verb"),
    ("mow the lawn", "verb"), ("call the bank", "verb"), ("order cat food", "verb"),
    ("pick up 2 pizzas", "verb"), ("feed the sourdough starter", "verb"),
    ("renew my library card", "verb"), ("take my vitamins", "verb"), ("stretch", "verb"),
    ("go for a run", "verb"), ("review the pull request", "verb"), ("write thank you notes", "verb"),
]

HELDOUT_TITLES = [
    ("orthodontist visit", "noun"), ("brunch with Alex", "noun"), ("board meeting", "noun"),
    ("pilates", "noun"), ("massage", "noun"), ("violin lesson", "noun"),
    ("supper with the Parkers", "noun"), ("checkup", "noun"), ("poker night", "noun"),
    ("standing desk delivery", "noun"), ("roadmap review", "noun"), ("tea with Nana", "noun"),
    ("school concert", "noun"), ("climbing session", "noun"), ("garden club", "noun"),
    ("mortgage signing", "noun"), ("kickboxing", "noun"), ("lunch with Tomas", "noun"),
    ("recruiter call", "noun"), ("trivia night", "noun"), ("workshop prep", "noun"),
    ("call the electrician", "verb"), ("fold the laundry", "verb"), ("book flights", "verb"),
    ("send the invoice", "verb"), ("water the tomatoes", "verb"), ("change the air filter", "verb"),
    ("call Aunt Rosa", "verb"), ("buy 3 gift cards", "verb"), ("refill my prescription", "verb"),
    ("update the website", "verb"), ("take the car in", "verb"),
]

# ------------------------------------------------------------------ locations: (preposition, name)

TRAIN_LOCATIONS = [
    ("at", "the library"), ("at", "Joe's Cafe"), ("in", "Room 204"), ("at", "the gym"),
    ("in", "the conference room"), ("at", "123 Main St"), ("at", "Dr. Patel's office"),
    ("on", "Zoom"), ("at", "the community center"), ("at", "Luigi's"), ("in", "the kitchen"),
    ("at", "Central Park"), ("at", "the office"), ("in", "Building C"), ("at", "Mom's house"),
    ("at", "the dog park"), ("at", "the Hilton downtown"), ("at", "Studio 5"),
    ("on", "Google Meet"), ("at", "the train station"),
]

HELDOUT_LOCATIONS = [
    ("at", "the rec center"), ("at", "Blue Bottle"), ("in", "Room 12"), ("at", "city hall"),
    ("in", "the lobby"), ("at", "88 Oak Ave"), ("on", "Teams"), ("at", "the community garden"),
]

# ------------------------------------------------------------------ sentence frames: (pattern, kind)
# Slots: {title} natural text, {Title} capitalized, {date} / {on_date} (with "on" where natural),
# {time} ("at 3pm", or a range), {tbare} ("3pm"), {rec} recurrence phrase.

TRAIN_FRAMES = [
    ("{Title} {date} {time}", "any"),
    ("Schedule {title} {on_date} {time}", "any"),
    ("Add {title} to my calendar {on_date} {time}", "any"),
    ("Remind me to {title} {on_date} {time}", "verb"),
    ("{Title}, {date}, {tbare}", "any"),
    ("Put {title} on the calendar for {date} {time}", "any"),
    ("Can you set up {title} for {date} {time}?", "any"),
    ("Book {title} {on_date} {time}", "noun"),
    ("I have {title} {on_date} {time}", "noun"),
    ("New event: {title} {date} {time}", "any"),
    ("Calendar: {title} {date} {time}", "any"),
    ("Don't let me forget to {title} {on_date} {time}", "verb"),
    ("{Title} - {date} {time}", "any"),
    ("Please add {title} {on_date} {time}", "any"),
    ("{date} {time}: {title}", "any"),
    ("Set up {title} {on_date} {time}", "any"),
    ("Create an event for {title} {date} {time}", "any"),
    ("I need to {title} {on_date} {time}", "verb"),
    ("{Title} {on_date} {time}", "any"),
    ("Make a reminder to {title} {date} {time}", "verb"),
]

HELDOUT_FRAMES = [
    ("Please put {title} in for {date} {time}", "any"),
    ("{on_date} {time}, {title}", "any"),
    ("I'd like to schedule {title} {on_date} {time}", "any"),
    ("Mark my calendar: {title} {date} {time}", "any"),
    ("{Title} is {on_date} {time}", "noun"),
    ("Set a reminder to {title} {on_date} {time}", "verb"),
    ("Need {title} {on_date} {time}", "any"),
    ("Could you add an entry called {title} for {date} {time}?", "any"),
    ("Throw {title} on my schedule {date} {time}", "any"),
    ("Pencil in {title} {on_date} {time}", "any"),
]

# Repeating events: {rec} is the recurrence phrase, {time} the time. "starting ..." and
# "until ..." are appended after the frame.
TRAIN_REC_FRAMES = [
    ("{Title} {rec} {time}", "any"),
    ("Schedule {title} {rec} {time}", "any"),
    ("Set up {title} {rec} {time}", "any"),
    ("Remind me to {title} {rec} {time}", "verb"),
    ("Add {title} to my calendar {rec} {time}", "any"),
    ("I have {title} {rec} {time}", "noun"),
    ("Create a repeating event: {title} {rec} {time}", "any"),
    ("{rec} {time}: {title}", "any"),
    ("Book {title} {rec} {time}", "noun"),
    ("I need to {title} {rec} {time}", "verb"),
]

HELDOUT_REC_FRAMES = [
    ("Please set {title} to repeat {rec} {time}", "any"),
    ("Put {title} on my calendar {rec} {time}", "any"),
    ("Recurring: {title}, {rec}, {tbare}", "any"),
    ("Make {title} a regular thing, {rec} {time}", "any"),
    ("Could you make {title} repeat {rec} {time}?", "any"),
    ("I'd like {title} {rec} {time}", "noun"),
]

# Short frames for rejections; {when} holds whatever the request does (or doesn't) say about time
TRAIN_SHORT_FRAMES = [
    ("Remind me to {title} {when}", "verb"), ("{Title} {when}", "any"), ("Schedule {title} {when}", "any"),
    ("I have {title} {when}", "noun"), ("Add {title} {when}", "any"), ("Put {title} on my calendar {when}", "any"),
    ("Set up {title} {when}", "any"), ("{when}: {title}", "any"),
]
HELDOUT_SHORT_FRAMES = [
    ("Please put {title} in {when}", "any"), ("Mark my calendar for {title} {when}", "any"),
    ("Pencil in {title} {when}", "any"), ("I'd like {title} {when}", "noun"),
    ("Need to {title} {when}", "verb"), ("Create an event: {title} {when}", "any"),
]

# ------------------------------------------------------------------ rejections

# Requests that are not about creating an event (many mention dates/times on purpose)
TRAIN_OFFTOPIC = [
    "what's the weather tomorrow at 3pm?", "how many days until Christmas?",
    "convert 3pm PST to EST", "what day of the week is October 31st?", "tell me a joke",
    "cancel my 3pm meeting", "what's on my calendar tomorrow?", "who won the game last night?",
    "translate good morning into Spanish", "move my dentist appointment to Friday",
    "delete all my events", "what time is it in Tokyo?", "write me a poem about Mondays",
    "how long is a flight from Boston to Denver?", "is the library open on Sundays?",
    "what's 15% of 80?", "recommend a good podcast", "summarize my emails from yesterday",
    "when is the next full moon?", "how do I make sourdough starter?",
    "what time does the sun set today?", "rename my 2pm meeting",
    "remove the yoga class from Thursday", "what is a recurring event?",
    "set an alarm for 6am",
]

HELDOUT_OFFTOPIC = [
    "what's the capital of Australia?", "will it rain on Saturday?", "show me my schedule for next week",
    "reschedule lunch to 1pm", "how many weeks are left in the year?", "read my last text message",
    "what's the date three weeks from today?", "tell me about the Roman empire",
    "delete the standup on Monday", "play some jazz",
    "start a timer for 20 minutes", "what's a good name for a cat?",
    "what is the date next Friday?", "is tomorrow a holiday?", "how do I add a guest to a meeting?",
    "what meetings do I have on Wednesday?", "email Sam that I'm running late",
    "how many hours until midnight?", "give me a packing list for a beach trip",
    "when does daylight saving time end?", "remind me what I did last Tuesday",
    "find a free slot next week", "what's the weather on Sunday morning?",
    "convert 10am Eastern to Pacific",
]

# Vague or incomplete time/date wording that should NOT produce a call
VAGUE_WHEN = [
    "sometime next week", "sometime this month", "whenever works", "soon", "later",
    "tomorrow morning", "tomorrow afternoon", "Friday evening", "next week", "this weekend",
    "after work", "over the weekend", "sometime Thursday", "in the morning", "at some point today",
    "later this week", "early next month", "tonight",
]

PAST_WHEN = ["yesterday", "last Tuesday", "last week", "two days ago", "last Friday", "last night", "the day before yesterday"]

# Dates that do not exist
BAD_DATES = ["February 30", "April 31", "June 31", "November 31", "September 31", "Feb 30"]

REJECTION_TEXT = {
    "offtopic": ["I can only create calendar events. Tell me what to schedule and when.",
                 "That isn't an event to create. Describe an event with a date and time."],
    "notime": ["What time should it start? I need a date and a time to create the event.",
               "I need a start time to create this event."],
    "nodate": ["Which day is this on? I need a date and a time to create the event.",
               "I need a date as well as a time to create this event."],
    "vague": ["That's too vague. Give me a specific date and time.",
              "I need an exact date and time, not 'sometime'."],
    "ambiguous": ["Is that am or pm? Please give the time with am/pm or in 24-hour form.",
                  "I need to know whether that's morning or evening."],
    "past": ["That time has already passed. Give me a future date and time.",
             "I can only create events in the future."],
    "baddate": ["That date doesn't exist. Please check the date.",
                "There's no such date. Give me a valid date and time."],
}


# ------------------------------------------------------------------ train-only extensions (D10)
# More titles and sentence frames, used ONLY when generating train.jsonl. The eval and test "ind"
# share still draws from the TRAIN_* lists above, so eval_v1 / test_v1 stay byte-identical.
# Nothing here may overlap HELDOUT_* (tests/test_generate.py checks titles and frames).

def _compose_titles() -> list:
    """Titles built from small parts, a fixed sample per pattern so no pattern dominates."""
    import random
    rng = random.Random(10)
    people = ["Ben", "Chloe", "Dana", "Eli", "Farah", "Gus", "Hana", "Ivan", "Jules", "Kofi", "Lena",
              "Mateo", "Nora", "Omar", "Pia", "Quinn", "Ravi", "Sofia", "Theo", "Uma", "Uncle Mike",
              "Grandpa", "my sister", "my brother", "Coach Dale", "the Nguyens"]
    out = []

    def add(patterns, kind, n):
        texts = sorted(set(patterns))
        out.extend((t, kind) for t in rng.sample(texts, min(n, len(texts))))

    add([f"{m} with {p}" for m in ("lunch", "coffee", "dinner", "drinks", "breakfast", "a walk", "a catch-up")
         for p in people], "noun", 30)
    add([f"{a} {s}" for a in ("tennis", "ballet", "chess", "cooking", "painting", "Spanish", "drumming",
                              "salsa", "boxing", "rowing", "archery", "knitting", "judo")
         for s in ("lesson", "class", "practice", "club")], "noun", 30)
    add([f"{p} {w}" for p in ("dermatologist", "accountant", "chiropractor", "pediatrician", "audiologist",
                              "mechanic", "tailor", "optometrist", "podiatrist", "vet", "landlord")
         for w in ("appointment", "visit", "fitting", "meeting")], "noun", 20)
    add([f"{a} {b}" for a in ("marketing", "hiring", "vendor", "security", "product", "sales", "finance",
                              "design", "support", "ops", "legal")
         for b in ("sync", "review", "committee", "check-in", "standup", "offsite")], "noun", 25)
    add([f"{p} {w}" for p in ("wedding", "graduation", "retirement", "anniversary", "housewarming", "baby shower")
         for w in ("rehearsal", "party", "dinner", "planning")], "noun", 12)
    add([f"{v} {p}" for v in ("call", "email", "text", "message", "ring")
         for p in people], "verb", 25)
    add([f"{v} {n} {t}" for v in ("buy", "order", "pick up", "print", "return", "grab")
         for n in (2, 4, 6, 12, 20) for t in ("notebooks", "cupcakes", "copies of the agenda", "packages",
                                              "light bulbs", "tickets", "bagels", "name tags")], "verb", 25)
    add([f"{v} the {t}" for v in ("clean", "vacuum", "wash", "organize", "empty", "sweep")
         for t in ("kitchen", "bathroom", "fridge", "car", "stairs", "porch", "closet", "attic", "pantry")],
        "verb", 20)
    add([f"pay the {b}" for b in ("electric bill", "credit card", "parking ticket", "water bill", "tuition",
                                 "gym membership", "internet bill")] +
        [f"renew {d}" for d in ("my license", "the car registration", "my gym membership", "the lease",
                                "my parking permit", "my insurance")], "verb", 12)
    add([f"{v} the {t}" for v in ("share", "post", "finalize", "proofread", "print", "forward")
         for t in ("newsletter", "slides", "proposal", "agenda", "minutes", "contract", "flyer")], "verb", 15)
    add([f"{v} {t}" for v in ("replace", "restock", "reorder")
         for t in ("the bird feeder", "the printer paper", "the first aid kit", "the coffee beans",
                   "my contact lenses", "the dog food")], "verb", 10)
    return out


TRAIN_TITLES_EXTRA = _compose_titles() + [
    ("graduation ceremony", "noun"), ("open house", "noun"), ("book launch", "noun"),
    ("jury duty", "noun"), ("car service", "noun"), ("daycare pickup", "noun"), ("team offsite", "noun"),
    ("tennis match", "noun"), ("cooking with Grandma June", "noun"), ("volunteer shift", "noun"),
    ("blood donation", "noun"), ("flu shot", "noun"), ("school pickup", "noun"), ("piano recital", "noun"),
    ("parent night", "noun"), ("fantasy football draft", "noun"), ("karaoke night", "noun"),
    ("trail run", "noun"), ("beach day", "noun"), ("family photo shoot", "noun"),
    ("take the bins out", "verb"), ("take the dog to the vet", "verb"), ("update my resume", "verb"),
    ("update the budget sheet", "verb"), ("change the bed sheets", "verb"),
    ("change the smoke detector batteries", "verb"), ("reserve a table for 6", "verb"),
    ("call the insurance company", "verb"), ("call the school", "verb"), ("call the vet", "verb"),
    ("check the mail", "verb"), ("rotate the tires", "verb"), ("defrost the freezer", "verb"),
    ("meditate", "verb"), ("journal", "verb"), ("do the dishes", "verb"), ("study for the exam", "verb"),
    ("finish chapter 3", "verb"), ("submit the timesheet", "verb"), ("book a haircut", "verb"),
    ("sign the permission slip", "verb"), ("return the library books", "verb"),
]

# Sentence frames. None of these reuses a held-out lead-in (Throw, Pencil in, Mark my calendar,
# Please put ... in for, I'd like, Could you add an entry called, Need, Set a reminder to).
TRAIN_FRAMES_EXTRA = [
    ("Jot down {title} for {date} {time}", "any"),
    ("Slot in {title} {on_date} {time}", "any"),
    ("Log {title} {on_date} {time}", "any"),
    ("Block off time for {title} {on_date} {time}", "any"),
    ("Hey, can you add {title} {on_date} {time}?", "any"),
    ("I've got {title} {on_date} {time}", "noun"),
    ("Add a calendar entry for {title}, {date} {time}", "any"),
    ("Reserve {date} {time} for {title}", "any"),
    ("{Title} @ {date} {time}", "any"),
    ("Remember: {title} {on_date} {time}", "any"),
    ("Hold {on_date} {time} for {title}", "any"),
    ("Can you put {title} down for {date} {time}?", "any"),
    ("Don't forget {title} {on_date} {time}", "any"),
    ("Heads up: {title} {on_date} {time}", "any"),
    ("Get {title} on the calendar for {date} {time}", "any"),
    ("I want to {title} {on_date} {time}", "verb"),
    ("Time to {title} {on_date} {time}", "verb"),
    ("Ping me to {title} {on_date} {time}", "verb"),
    ("Appointment: {title} {date} {time}", "noun"),
    ("Enter {title} for {date} {time}", "any"),
    ("Pop {title} into my calendar {on_date} {time}", "any"),
    ("Yo, {title} {on_date} {time}", "any"),
    ("Make a note: {title} {on_date} {time}", "any"),
    ("Calendar entry - {title} - {date} {time}", "any"),
    ("Would you mind adding {title} {on_date} {time}?", "any"),
    ("Can you fit in {title} {on_date} {time}?", "any"),
    ("Add to calendar: {title} {date} {time}", "any"),
    ("Set {title} for {date} {time}", "any"),
    ("Lock in {title} {on_date} {time}", "any"),
    ("I'm planning {title} {on_date} {time}", "noun"),
    ("Sign me up for {title} {on_date} {time}", "noun"),
    ("Remind me about {title} {on_date} {time}", "noun"),
    ("{date} {time} - {title}", "any"),
    ("Got {title} {on_date} {time}", "noun"),
    ("Please schedule {title} for {date} {time}", "any"),
    ("Add {title} for me {on_date} {time}", "any"),
    ("Put a hold on the calendar for {title}, {date} {time}", "any"),
    ("Note to self: {title} {on_date} {time}", "any"),
]

TRAIN_REC_FRAMES_EXTRA = [
    ("Repeat {title} {rec} {time}", "any"),
    ("Add {title} as a repeating event, {rec} {time}", "any"),
    ("Keep {title} on my calendar {rec} {time}", "any"),
    ("{Title} should happen {rec} {time}", "noun"),
    ("Regular event - {title} {rec} {time}", "any"),
    ("Remember to {title} {rec} {time}", "verb"),
    ("Hold a slot for {title} {rec} {time}", "any"),
    ("Put {title} down {rec} {time}", "any"),
    ("Log {title} {rec} {time}", "any"),
    ("Ongoing: {title} {rec} {time}", "any"),
    ("Can you set {title} up {rec} {time}?", "any"),
    ("I do {title} {rec} {time}", "noun"),
    ("Ping me to {title} {rec} {time}", "verb"),
]

TRAIN_SHORT_FRAMES_EXTRA = [
    ("Jot down {title} {when}", "any"), ("Log {title} {when}", "any"),
    ("Hold time for {title} {when}", "any"), ("Could {title} go on the calendar {when}?", "any"),
    ("{Title}, {when}", "any"), ("Add a note for {title} {when}", "any"),
    ("Enter {title} {when}", "any"), ("Get {title} on the books {when}", "any"),
    ("Plan on {title} {when}", "any"), ("Don't forget to {title} {when}", "verb"),
    ("Remember to {title} {when}", "verb"), ("Can you fit in {title} {when}?", "any"),
]
