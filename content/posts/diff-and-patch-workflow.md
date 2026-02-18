---
title: "From `diff` to `patch`: a tiny but powerful GitHub-era workflow"
date: 2026-01-18
tags: ["git", "github", "unix", "diff", "patch"]
---

Today we’re starting our GitHub learning journey, but before we jump into pull requests and fancy GUIs, I want you to get comfortable with a _super old-school_ (and still extremely useful) workflow:

These code and idea come from the course on coursera "Introduction to Git and GitHub"

```bash
cat
diff rearrange1.py rearrange2.py
cat cpu_usage.diff
patch cpu_usage.py < cpu_usage.diff
```

If you can read and understand those four lines, you’ll _never_ feel lost when someone says “here’s a patch” or when a tool spits out a `.diff` file.

---

## 1) `cat` — “show me the file (or whatever is piped in)”

`cat` is short for **concatenate**, but in day-to-day work we mostly use it as “print the file to the terminal.”

### Common uses

```bash
cat cpu_usage.diff
cat cpu_usage.py
```

That’s just: “show me what’s inside.”

You can also use it in pipelines, but for our learning goal, treat it as your quick “peek” command.

---

## 2) `diff rearrange1.py rearrange2.py` — “what changed?”

`diff` compares two files line-by-line and prints the differences.

```bash
diff rearrange1.py rearrange2.py
```

Meaning: compare **rearrange1.py** (old) vs **rearrange2.py** (new).

### What does the output look like?

There are a few diff formats. Plain `diff` often looks like this:

- Lines with `<` come from the _first_ file
- Lines with `>` come from the _second_ file
- A line like `3c3` or `10,12d9` is diff’s way of saying _where_ the change happened

Example (simplified):

```diff
5c5
< print("old message")
---
> print("new message")
```

That means line 5 changed.

### Why this matters for GitHub

Git is basically a giant diff engine with memory.  
When you run `git diff`, you’re doing the same mental skill: _reading what changed_.

So learning `diff` now is like learning to read the “DNA” of Git.

---

## 3) `cat cpu_usage.diff` — “inspect the patch before you apply it”

A `.diff` file is usually just text that describes changes.

```bash
cat cpu_usage.diff
```

This is a habit I strongly recommend:

> **Always read a patch before applying it**, especially if you didn’t generate it yourself.

You’re a data scientist. You wouldn’t run a random notebook you found online without scanning it first. Same energy.

---

## 4) `patch cpu_usage.py < cpu_usage.diff` — “apply the changes”

Now the cool part:

```bash
patch cpu_usage.py < cpu_usage.diff
```

This means:

- run the `patch` program
- tell it the _target file_ is `cpu_usage.py`
- feed it the diff content using input redirection `< cpu_usage.diff`

In other words:

> “Take the instructions in `cpu_usage.diff` and modify `cpu_usage.py` accordingly.”

### What patch does under the hood

A diff contains _context lines_ (unchanged nearby lines). `patch` uses those to find where changes belong.

That’s why patches can sometimes apply even if line numbers shifted slightly — as long as the surrounding context still matches.

### What you’ll see when it works

Typical successful output looks like:

- `patching file cpu_usage.py`

If it can’t find where to apply the change, you might see:

- `Hunk #1 FAILED at ...`
- and it may create a `.rej` file with the failed chunk

If that happens, it’s not the end of the world — it just means the file has drifted too much from what the patch expected.

---

## A practical mini-workflow (what you should do every time)

Here’s a clean “teacher-approved” sequence:

1. **Compare** two versions:

   ```bash
   diff rearrange1.py rearrange2.py
   ```

2. **Save** the diff (optional but common):

   ```bash
   diff rearrange1.py rearrange2.py > change.diff
   ```

3. **Inspect** the diff:

   ```bash
   cat change.diff
   ```

4. **Apply** it to a file:

   ```bash
   patch some_file.py < change.diff
   ```

5. **Verify**:
   ```bash
   python -m py_compile some_file.py
   # or run your tests / run the script
   ```

That last step is important: patch is mechanical—it will happily apply changes that break your code.

---

## How this connects to GitHub (without jumping ahead)

On GitHub, changes usually show up as:

- a commit (which is basically a saved diff)
- a pull request (a pile of diffs + discussion)
- or a `.patch` / `.diff` file someone attaches

So even when you’re living in `git`, knowing the classic tools gives you confidence.

### Bonus: `git` equivalents you’ll meet soon

- `diff` → `git diff`
- `patch` → `git apply` (applies a patch) or `git am` (applies patch _as commits_)

But don’t worry about those yet—the mental model is the same.

---

## Common mistakes (so you don’t waste an hour)

### 1) Applying a patch to the wrong file

If the diff was created for `cpu_usage.py`, but you run patch against a different file name, you’ll get failures.

### 2) Patch was generated in a different directory structure

Sometimes diffs contain paths like `a/cpu_usage.py` and `b/cpu_usage.py` (common with Git).  
Then `patch` may need a strip level:

```bash
patch -p1 < cpu_usage.diff
```

You’ll learn this quickly once you see it in the wild. Just remember `-p1` is for “drop the first path component.”

### 3) Not checking the result

Always open the file or run a quick check after patching. Even better: use Git so you can easily undo.

---

In real projects you’ll encounter situations like:

- someone sends a `.diff` in Slack
- you’re on a server without a GUI
- you need to review changes fast
- you want to apply a fix without copy-pasting code blocks

This workflow is lightweight, universal, and works basically everywhere.
