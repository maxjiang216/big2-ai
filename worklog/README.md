# Worklog

A makeshift issue tracker: nested folders of markdown, versioned with the code it
describes. It exists to hold the things a commit message cannot — why an approach was
chosen over the one you would expect, what was tried and abandoned, what the next person
(or the next session) should pick up.

## Structure

```
worklog/
  YYYY-MM-DD-what-were-doing/          an epic, dated from the day it started
    YYYY-MM-DD-NN-name.md              a session's work, or a self-contained task
    YYYY-MM-DD-bigger-task/            a task that needs several files or days
      YYYY-MM-DD-NN-name.md
```

An **epic** is a folder, named for the day it started and what it is about. Anything
inside it is either a single markdown file — one session, or one task — or a folder of
them when a task runs long enough to need its own days.

Dates are the date of the work, not the date of writing. They sort chronologically, which
is the only ordering that matters here.

**Entry files carry a two-digit sequence number**, `YYYY-MM-DD-NN-name.md`, counting from
01 within a day. More than one session lands on the same date often enough that a bare
date stops sorting them, and renaming an entry after the fact breaks any link to it.
Folders do not take a number: an epic or a multi-day task spans dates, so a sequence
within one date would mean nothing.

## `ENGINE-CHECKLIST.md`

One file at the top of `worklog/`, not inside an epic, because it outlives them. It
collects everything the engine does that **`RULES.md` does not settle** — a case the rules
are silent on, a behaviour derived from a proof rather than a quotation, an edge case rare
enough that the implementation has never been exercised.

`RULES.md` is the human-readable variant description and `src/core/` is the authoritative
implementation. Where those two are both silent and the code still had to choose, the
choice belongs here. Add to it whenever a fix rests on an inference rather than a
quotation, and say *what to watch for* rather than only what was assumed. Tick items off
as a test or a real game confirms them.

## What goes in an entry

- **TL;DR** — a list of the major things done, one line each. Written so that reading only
  this tells you whether the entry is worth opening.
- **Details** — one section per item: what changed, which commits, which branches, and
  the reasoning that is not visible in the diff. `RULES.md` citations belong here; so do
  the alternatives that were rejected and why.
- **Carried forward** — the previous entry's open items, each either struck through with
  what closed it or restated as still live. See *Sweeping* below.
- **Next steps** — what is left, what is blocked on what, and any decision waiting on a
  human. Be specific enough that it can be picked up cold.

The bias is toward writing down what you would otherwise have to re-derive: a rule read
three times before it made sense, a measurement and its resolution, a design choice whose
alternative still looks tempting.

## Every fix carries a regression test

A bug that was found once and fixed silently will be found again. An entry that reports a
fix names the test that covers it, and that test must **fail before the fix and pass
after** — a test written against already-fixed code proves nothing about the bug it claims
to guard.

Where the bug survived because an existing suite had a hole, say what the hole was. The
hole is the more useful finding: `test/test_legal_moves.cpp` containing zero occurrences
of the word "bomb" explains a whole class of missed defects, where "the bomb comparison
was wrong" explains one.

Tests live where the code does — `test/*.cpp` for the C++ engine (registered in
`test/test_main.cpp`, run by CI), `nn/test_*.py` for the training pipeline, `web/test_*.mjs`
for the browser port. If a fix lands in a layer with no suite, that gap is itself a
next-step item.

## Sweeping

**Every entry ends by sweeping the previous one's open items**, not by starting a fresh
list. Without this, next-steps sections accumulate: the same four items get restated in
four entries, two of them already done, and nobody can tell which list is current.

The sweep is a *Carried forward* section with two parts:

- **Done since** — struck through, each with what closed it. Keeping the strikethrough
  rather than deleting the line means the next reader can see the item was considered and
  resolved, not dropped.
- **Still live** — restated, not linked to. An item that survives three sweeps unchanged is
  worth a sentence about *why* it keeps surviving; "the bomb-response regression has not
  been retrained through" reads differently on its fifth appearance than its first.

Sweep only the entry before yours. It has already swept the one before it, so the chain
carries everything forward without anyone re-reading the whole epic. If an item belongs to
the engine rather than to a session — an inference, an unexercised edge case, a divergence
between two implementations — move it to `ENGINE-CHECKLIST.md` instead and stop carrying it.
