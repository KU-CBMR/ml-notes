---
title: "Reading History, Staging Like a Pro, and Not Nuking Your Repo: a practical Git toolbox"
date: 2025-12-20
tags: ["git", "github", "diff", "log", "branch", "merge", "revert", "reset"]
---

You’ve already got the core loop down: **edit → add → commit**.  
Now we’re going to level up with the “daily driver” commands:

```bash
git log -p
git log --stat
git add -p
git diff
git diff --staged
git rm process.py
git mv disk_usage.py check_free_space.py
git add *

git reset HEAD output.txt
git revert HEAD
git revert 30e7

git log -p -2
git log --graph --oneline

git show
git show 30e7071288226XXXXX
git show 30

git branch new-feature
git branch -d new-feature
git branch -D <name>

git checkout new-feature
git checkout -b even-better-feature
git merge even-better-feature
git merge --abort
```

Same deal as before: what it does, why it exists, and how to use it without pain.

---

# 1) Reading commit history like a detective

## `git log -p` — show commits _plus the patch_

```bash
git log -p
```

This prints commit messages and the actual diff for each commit (the patch).

When you’re debugging a bug, this is gold:

- “When did this line change?”
- “Who changed it?”
- “What exactly changed?”

**Tip:** add `-2` or `-5` so you don’t drown in output.

---

## `git log --stat` — show file-level summary

```bash
git log --stat
```

Instead of full diffs, you get a summary like:

- which files changed
- how many insertions/deletions

This is perfect when you’re scanning history quickly.

---

## `git log -p -2` — last 2 commits with patches

```bash
git log -p -2
```

My everyday “show me what just happened” command.

---

## `git log --graph --oneline` — the “pretty branch map”

```bash
git log --graph --oneline
```

This draws an ASCII graph of branches and merges, with short one-line commits. Great for seeing whether your history is clean or chaotic.

---

# 2) Diffing like you mean it

## `git diff` — what changed but is NOT staged

```bash
git diff
```

This compares:

- **working directory** vs **staging area**

Translation: “What have I edited that I haven’t added yet?”

---

## `git diff --staged` — what’s staged but NOT committed

```bash
git diff --staged
```

This compares:

- **staging area** vs **last commit (HEAD)**

Translation: “What will go into my next commit if I commit now?”

If you learn only one “anti-regret” habit, it’s this:

> Always run `git diff --staged` before you commit.

---

# 3) Staging with precision

## `git add -p` — interactive staging (the grown-up version of `git add`)

```bash
git add -p
```

This lets you stage changes **hunk by hunk** (chunk by chunk). You’ll be prompted:

- stage this hunk? `y/n`
- split it? `s`
- edit it manually? `e`

### Why this is so valuable

Say you:

- fixed punctuation
- also added debug prints
- also changed a function name

Instead of one messy commit, you can stage only the punctuation changes and commit them separately.

Clean history = easier PR review = easier debugging = happier you.

---

# 4) Removing and renaming files (the Git-friendly way)

## `git rm process.py` — delete a file and stage the deletion

```bash
git rm process.py
```

This removes the file from disk _and_ stages that deletion, so your next commit records “file removed.”

If you manually delete the file with `rm process.py`, Git can still detect it, but `git rm` is the straightforward way.

---

## `git mv disk_usage.py check_free_space.py` — rename (or move) a file

```bash
git mv disk_usage.py check_free_space.py
```

This is basically:

- rename the file
- stage the change

Git doesn’t store “renames” as a special object, but it’s very good at detecting them. Using `git mv` just keeps things tidy and obvious.

---

## `git add *` — stage “everything matched by shell glob”

```bash
git add *
```

This stages all files matched by `*` **in your current directory** (and it depends on your shell). It’s not “wrong,” but I prefer safer commands like:

- `git add .` (stage changes in current directory, including new files)
- or explicit file paths

**Why caution matters:** `*` can behave differently than you expect (and it’s easy to accidentally stage generated junk).

---

# 5) Undoing safely: `reset` vs `revert`

These two words sound similar but behave very differently.

## `git reset HEAD output.txt` — unstage a file (keep your edits)

```bash
git reset HEAD output.txt
```

This is the classic “oops, I staged the wrong file.”

It moves `output.txt` from:

- **staged** → back to **modified/not staged**

Your file contents stay exactly as-is on disk. You’re not deleting your work, just removing it from the staging area.

---

## `git revert HEAD` — undo a commit by making a new commit

```bash
git revert HEAD
```

This does **not** rewrite history. It creates a new commit that reverses the changes introduced by `HEAD`.

This is the “team-safe” undo:

- great when you already pushed
- great when others might be based on your commits

---

## `git revert 30e7` — revert a specific commit (by short SHA)

```bash
git revert 30e7
```

Same idea, but targets a particular commit. Git accepts short SHAs as long as they’re unambiguous.

---

# 6) Inspecting a specific commit or object

## `git show` — show the latest commit (by default)

```bash
git show
```

This usually shows:

- the most recent commit
- its message
- its patch (diff)

It’s like `git log -p -1` in spirit.

---

## `git show 30e7071288226XXXXX` — show a specific commit by SHA

```bash
git show 30e7071288226XXXXX
```

This displays that exact commit. (Your real SHA will be a full hex string; the `XXXXX` here is just placeholder.)

---

## `git show 30` — short SHA works too (if unique)

```bash
git show 30
```

If multiple commits start with `30`, Git will complain and ask for a longer prefix.

---

# 7) Branching: creating, deleting, and switching

## `git branch new-feature` — create a branch pointer

```bash
git branch new-feature
```

This creates a branch name pointing to your current commit. It does **not** switch you to it.

A branch is basically: “a named pointer to a commit.”

---

## `git branch -d new-feature` — delete a branch (safe)

```bash
git branch -d new-feature
```

This only deletes the branch if Git believes it’s already merged (so you don’t accidentally lose work).

---

## `git branch -D <name>` — delete a branch (force)

```bash
git branch -D <name>
```

This deletes it even if it contains unmerged commits. Use when you’re absolutely sure—or when you’re cleaning up a branch you don’t care about.

---

## `git checkout new-feature` — switch branches

```bash
git checkout new-feature
```

This updates your working directory to match that branch.

Modern Git also has `git switch`, but `checkout` is still everywhere and worth knowing.

---

## `git checkout -b even-better-feature` — create + switch in one go

```bash
git checkout -b even-better-feature
```

This is the “start a new branch to try something” move. You’ll use this constantly.

---

# 8) Merging (and escaping a merge)

## `git merge even-better-feature` — merge a branch into the current branch

```bash
git merge even-better-feature
```

Important detail: merges always merge **into your current branch**.

So if you are on `main` and you run that, you’re merging `even-better-feature` into `main`.

If the histories are compatible, Git auto-merges. If not, you get conflicts.

---

## `git merge --abort` — bail out of a conflicted merge

```bash
git merge --abort
```

If you started a merge and it went sideways (conflicts, panic, wrong branch), `--abort` tries to restore the state from before you started merging.

This is your “undo the merge attempt” parachute.

---

# A mini scenario that uses _almost all_ of these commands

Let’s pretend you want to rename `disk_usage.py` and remove an unused script:

1. Create a branch and switch to it:

```bash
git checkout -b even-better-feature
```

2. Rename the file and remove an old one:

```bash
git mv disk_usage.py check_free_space.py
git rm process.py
```

3. Stage carefully and review:

```bash
git status
git diff --staged
```

4. Commit:

```bash
git commit -m "Rename disk usage script and remove unused process script"
```

5. Go back to main and merge:

```bash
git checkout main
git merge even-better-feature
```

6. If something explodes during merge:

```bash
git merge --abort
```

---

# The “don’t get burned” cheat sheet

- **See what you changed (not staged):** `git diff`
- **See what you’re about to commit:** `git diff --staged`
- **Stage carefully:** `git add -p`
- **Unstage a file:** `git reset HEAD <file>`
- **Undo a commit safely (especially after pushing):** `git revert <commit>`
- **See commit content fast:** `git show` / `git show <sha>`
- **Make branches cheap and fearless:** `git checkout -b <branch>`
- **Merge into current branch:** `git merge <branch>`
- **Abort a messy merge:** `git merge --abort`

---
