---
title: "Going Remote: pull, push, fetch, remotes, and a gentle intro to rebase"
date: 2025-12-21
tags:
  ["git", "github", "remote", "push", "pull", "fetch", "rebase", "credentials"]
---

So far we’ve been living happily on our laptop: commits, diffs, branches.  
Now it’s time to connect your repo to the outside world (GitHub, GitLab, a company server… same idea).

Here are some commands:

```bash
git config --global credential.helper cache
git pull
git push
git push --delete origin refactor
git push -u origin refactor
git remote -v
git remote show origin
git remote update
git remote
git rebase master
git rebase --continue
git rebase -i master
git fetch
git branch -r

git log --graph --oneline --all
```

These code and idea come from the course on coursera "Introduction to Git and GitHub"

---

# 1) Credential caching (so you don’t type secrets every 2 minutes)

## `git config --global credential.helper cache`

```bash
git config --global credential.helper cache
```

This tells Git to **temporarily cache your credentials** so you’re not prompted constantly.

A few notes (because this matters in real life):

- This is most common when you use **HTTPS remotes**
- It caches credentials in memory for a limited time (default is often ~15 minutes, but can vary)
- It’s convenient, but not the same as a proper credential manager

If you’re using **SSH remotes** (the `git@github.com:...` style), credential caching is usually handled by your SSH agent instead.

---

# 2) Understanding “remote” in one sentence

A **remote** is just a nickname pointing to a repository URL.  
Most repos have a remote named **`origin`**.

When people say “push to GitHub,” what they mean is:

> push to the remote named `origin`, which happens to point to GitHub.

---

# 3) Seeing your remotes

## `git remote`

```bash
git remote
```

Lists remote names (usually just `origin`).

## `git remote -v`

```bash
git remote -v
```

Shows the remote URLs for fetch and push, like:

- `origin  git@github.com:you/repo.git (fetch)`
- `origin  git@github.com:you/repo.git (push)`

This is the first thing I check when someone says, “Why am I pushing to the wrong place?”

## `git remote show origin`

```bash
git remote show origin
```

This is the “details page”:

- what branches exist on the remote
- which local branches track which remote branches
- whether anything is ahead/behind
- remote HEAD branch

Super useful when you inherit a repo and you’re like, “What is even going on here?”

---

# 4) Fetch vs pull (this distinction saves careers)

## `git fetch`

```bash
git fetch
```

Fetch downloads updates from the remote **without changing your working branch**.

Think of it as: “Update my knowledge of what the remote looks like.”

It updates remote-tracking branches like `origin/main`, `origin/refactor`, etc.

## `git pull`

```bash
git pull
```

Pull is basically:

> `git fetch` + integrate (merge or rebase)

So `git pull` _does_ modify your current branch (because it integrates remote changes).

### Rule of thumb

- If you want to look first: use `fetch` then inspect.
- If you’re confident: use `pull`.

---

# 5) Remote branches

## `git branch -r`

```bash
git branch -r
```

Lists remote-tracking branches, like:

- `origin/main`
- `origin/refactor`

These are Git’s local references to what the remote had the last time you fetched.

---

# 6) Pushing (sending your commits to the remote)

## `git push`

```bash
git push
```

This pushes your current branch commits to its upstream branch (if configured).

If no upstream is set, Git will usually tell you what to do next.

## `git push -u origin refactor`

```bash
git push -u origin refactor
```

This is the “first push of a new branch” command.

- `origin` is the remote name
- `refactor` is the branch you’re pushing
- `-u` sets the upstream tracking relationship

After this, you can usually just run `git push` from that branch without extra arguments.

## `git push --delete origin refactor`

```bash
git push --delete origin refactor
```

Deletes the branch **on the remote** (not your local branch).

This is what you do after a PR is merged and you want to clean up remote branches.

---

# 7) Keeping remote info fresh

## `git remote update`

```bash
git remote update
```

Fetches updates from remotes (similar to `fetch`, but it can update multiple remotes if you have more than one). It’s basically “refresh my remote knowledge.”

In many everyday workflows, `git fetch` is the more common go-to, but `remote update` is handy when you’re managing multiple remotes.

---

# 8) Rebase (the clean-history tool, used carefully)

You listed:

```bash
git rebase master
git rebase --continue
```

Two big notes:

1. Many repos now use `main` instead of `master` as the default branch name. If your mainline branch is `main`, you’d do `git rebase main` instead.
2. Rebase rewrites commit history (locally). That’s fine on your own branches, but it needs care once things are shared.

## `git rebase master` — replay your branch on top of master

```bash
git rebase master
```

Imagine you created a branch `refactor` from `master`, then `master` moved forward.

Rebase says:

> “Take my branch commits and replay them as if I started from the latest `master`.”

This often produces a cleaner, linear history compared to merging.

### Typical workflow (clean and safe)

From your feature branch:

```bash
git fetch
git rebase origin/master
```

(or `origin/main` depending on the repo)

Rebasing onto the remote-tracking branch avoids rebasing onto stale local info.

---

## Conflicts during rebase: `git rebase --continue`

If Git hits a conflict, it pauses and tells you what files need attention.

The flow is:

1. Fix conflicts in files
2. Stage the fixed files:
   ```bash
   git add <conflicted_files>
   ```
3. Continue rebase:
   ```bash
   git rebase --continue
   ```

If you realize you’re rebasing the wrong thing, you can usually abort with:

```bash
git rebase --abort
```

---

# 9) Visualizing everything: local + remote + branches

## `git log --graph --oneline --all`

```bash
git log --graph --oneline --all
```

This is the “show me the whole story” view:

- all branches (local + remote-tracking)
- a nice commit graph
- short one-line messages

If you ever feel lost, this command is your map.

---

# A realistic example: a clean feature-branch sync + push

Let’s say you’re on `refactor`.

1. Make sure you have the latest remote info:

```bash
git fetch
```

2. Rebase your branch onto the updated mainline:

```bash
git rebase origin/master
```

3. Push your branch:

```bash
git push
```

If you rewrote history via rebase and the remote already has your old commits, Git may reject the push.  
In that situation, you may need:

```bash
git push --force-with-lease
```

I’m mentioning this carefully because it’s powerful: `--force-with-lease` is the safer “force push” that refuses if the remote moved unexpectedly.

Use it only when you understand why the push was rejected (usually: you rebased a branch that already exists on the remote).

---

# The “what to use when” quick guide

- **Check remotes:** `git remote -v`
- **See remote details:** `git remote show origin`
- **Download remote updates without changing your branch:** `git fetch`
- **Download + integrate:** `git pull`
- **Publish commits:** `git push`
- **First push of a new branch:** `git push -u origin <branch>`
- **Delete remote branch:** `git push --delete origin <branch>`
- **Clean up / linearize history on your feature branch:** `git rebase <mainline>`
- **Continue after fixing conflicts:** `git rebase --continue`
- **See the whole graph:** `git log --graph --oneline --all`

---
