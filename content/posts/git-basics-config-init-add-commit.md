---
title: "Your First Real Git Repo: config, init, add, status, commit (and what `.git/` is really doing)"
date: 2025-12-19
tags: ["git", "github", "cli", "version-control", "workflow"]
---

Last time we played with `diff` and `patch`. Today we’re stepping into **actual Git**, using the below commands:

```bash
ls -la
ls -l .git/

git config --global user.email "me@example.com"
git config --global user.name "My name"
git config -l

git init
git add disk_usage.py
git status
git commit -m 'Add periods to the end of sentences.'
```

These code and idea come from the course on coursera "Introduction to Git and GitHub"

I’m going to explain what each one does, _why it exists_, and what to look for in the output—because Git is 10x easier when you know what it’s storing under the hood.

---

## 1) `ls -la` — “show me what’s in this folder (including hidden stuff)”

```bash
ls -la
```

- `ls` lists files
- `-l` shows the “long” format (permissions, size, timestamps)
- `-a` includes **hidden files** (anything starting with `.`)

This is the simplest way to sanity-check: _Where am I? What files exist?_  
Before Git, after Git, always good.

---

## 2) `ls -l .git/` — “what’s in the Git database?”

```bash
ls -l .git/
```

The `.git/` directory is the **entire repository metadata**. If you delete it, the folder is no longer a Git repo (you’ll still have your code, but Git “memory” is gone).

Inside `.git/` you’ll see stuff like:

- `HEAD` — points to the current branch / commit
- `config` — repo-specific configuration
- `objects/` — where Git stores _all content_ (commits, trees, blobs)
- `refs/` — pointers (branches/tags) to commits
- `index` — the staging area (more on that in a second)
- `hooks/` — optional scripts that run on events like commit

Think of `.git/` as Git’s private brain.

> If you understand `.git/` is “the database,” a lot of Git stops feeling magical.

---

## 3) `git config --global ...` — “tell Git who you are (for commits)”

```bash
git config --global user.email "me@example.com"
git config --global user.name "My name"
```

Every commit records an **author name + email**. Git doesn’t guess—**you must set it**.

- `--global` means “apply this for _all_ repos on this machine”
- without `--global`, it only affects the current repo

### Why Git cares

Git commits are meant to be traceable. On a team, you want to know:

- Who changed this?
- When?
- Why?

The author info is part of that story.

---

## 4) `git config -l` — “show me my current config”

```bash
git config -l
```

This prints config values Git is using (from system, global, and local scope). You’ll usually see your `user.name` and `user.email` in there.

Tip: if something behaves weirdly in Git, `git config -l` is a good first debugging move.

---

## 5) `git init` — “turn this folder into a Git repo”

```bash
git init
```

This creates the `.git/` directory and sets up the minimal structure Git needs.

After `git init`, your project directory becomes a Git repository.

### What _doesn’t_ happen

- Git does **not** automatically track files
- Git does **not** make a commit
- Git does **not** connect to GitHub yet

It just says: “Ok, I’m ready to start versioning.”

---

## 6) `git add disk_usage.py` — “put this file into the staging area”

```bash
git add disk_usage.py
```

This is the command most beginners misunderstand, so let’s make it crystal clear:

Git has **three** “places” your code can live:

1. **Working directory** (your actual files on disk)
2. **Staging area** (a.k.a. the “index”)
3. **Commit history** (permanent snapshots)

`git add` moves changes from **(1) working directory** into **(2) staging area**.

### The staging area is a feature, not a bug

It lets you craft commits intentionally.

Example: you changed 5 things, but you only want to commit 2 of them.  
You stage only what belongs together.

---

## 7) `git status` — “tell me what Git thinks is going on”

```bash
git status
```

This is your dashboard.

It will typically show:

- which branch you’re on
- what’s staged (ready to commit)
- what’s modified but not staged
- what’s untracked

If you run just one Git command all day, `git status` is a strong contender.

---

## 8) `git commit -m '...'` — “save a snapshot with a message”

```bash
git commit -m 'Add periods to the end of sentences.'
```

This creates a **commit** from whatever is currently in the staging area.

Two key things happen:

1. Git stores the file contents (as objects) in `.git/objects/`
2. Git creates a commit object that points to a tree (directory snapshot) + parent commit + metadata

### Why the message matters

A commit message is the “why” that future you will beg for.

This one is actually a decent message because it says what you changed.
Even better messages add _context_:

- What motivated the change?
- Any user impact?
- Any follow-up needed?

But for now: short and clear is perfect.

---

## Putting it all together: a tiny real example

Let’s say `disk_usage.py` prints some human-readable messages and you notice punctuation is inconsistent.

### Before

```python
print("Scanning disks")
print("Done")
```

You update it to:

```python
print("Scanning disks.")
print("Done.")
```

Now the Git flow:

```bash
git init
git add disk_usage.py
git commit -m "Add periods to the end of sentences."
```

That’s one clean, single-purpose commit.

And later, if you change formatting again, you’ll make another commit.  
That’s how Git turns messy iteration into a readable history.

---

## A few common “first day Git” mistakes

### 1) “I ran `git init`, why isn’t anything tracked?”

Because Git doesn’t track files automatically. You need `git add`.

### 2) “I edited the file, why didn’t `git commit` include it?”

Because the edit wasn’t staged. Run `git add` again.

### 3) “What’s the point of staging?”

It helps you make **clean commits**. Clean commits make PR reviews easier, debugging easier, and collaboration less painful.

---

## Quick habit to build (seriously worth it)

After any meaningful change:

```bash
git status
git add <files>
git status
git commit -m "Explain the change"
```

If you do that consistently, Git becomes boring—in the best way.

---

## What’s next

Once you can do `init → add → status → commit` comfortably, the next GitHub step is:

- create a remote
- push your branch
- open a pull request

We’ll get there. For now, make these commands feel like muscle memory.
