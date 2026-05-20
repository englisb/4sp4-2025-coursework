# 4sp4-2025-coursework

This repository is a combined snapshot repository. It stores the latest imported contents from multiple coursework repositories in subfolders, without preserving source commit history.

## Folder mapping

- `lab01/` ← `englisb/lab-01-4sp4-2025-13`
- `lab02/` ← `englisb/lab-02-4sp4-2025-13`
- `lab03/` ← `englisb/lab-03-4sp4-2025-13`
- `lab04/` ← `englisb/lab-04-4sp4-2025-13`
- `tut02/` ← `englisb/tutorial-02-4sp4-2025-englisb`
- `project/` ← `englisb/project-4sp4-2025-13`

## Automated import workflow

Run **Actions → Import snapshots → Run workflow** to import fresh snapshots into the folders above.

The workflow:
- clones each source repository from `main` using `secrets.IMPORT_TOKEN`
- deletes existing destination folder contents before copying (idempotent reruns)
- copies files without preserving git history
- commits and pushes updates to `main`

## Required secret: `IMPORT_TOKEN`

Create a fine-grained personal access token and save it as a repository Actions secret named `IMPORT_TOKEN`.

Minimum required access:
- **Source repositories** (`englisb/lab-01-4sp4-2025-13`, `englisb/lab-02-4sp4-2025-13`, `englisb/lab-03-4sp4-2025-13`, `englisb/lab-04-4sp4-2025-13`, `englisb/tutorial-02-4sp4-2025-englisb`, `englisb/project-4sp4-2025-13`): repository **Contents: Read**
- **Target repository** (`englisb/4sp4-2025-coursework`): repository **Contents: Read and write**

Add the secret in:
**Repository Settings → Secrets and variables → Actions → New repository secret**
