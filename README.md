# binder-notebooks

This repository contains lesson notebooks, theory, and quizzes that can be opened in Binder.

## Quick start (this project setup)

Use this Binder link to start the `binder-env` image and then pull this repository:

- https://mybinder.org/v2/gh/SigmoidAI/binder-env/main?urlpath=git-pull?repo=https://github.com/SigmoidAI/binder-notebooks.git

If the above link fails in some browsers, use the URL-encoded variant:

- https://mybinder.org/v2/gh/SigmoidAI/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps://github.com/SigmoidAI/binder-notebooks.git

## Binder URL structure

General format:

`https://mybinder.org/v2/gh/<owner>/<repo>/<ref>?urlpath=<target>`

- `<owner>/<repo>/<ref>`: the repository Binder builds (environment and startup files).
- `urlpath=<target>`: where Binder sends the user after launch.

In this project:

- Build source: `SigmoidAI/binder-env` on `main`
- Pulled content: `SigmoidAI/binder-notebooks`

### Why `git-pull` is used here

`binder-env` provides the runtime environment; `git-pull` fetches this notebooks repo into the live Binder session. This lets you keep environment and course content in separate repositories.

## Common Binder link patterns

### 1) Open JupyterLab root

`https://mybinder.org/v2/gh/<owner>/<repo>/<ref>?urlpath=lab`

### 2) Open classic notebook tree

`https://mybinder.org/v2/gh/<owner>/<repo>/<ref>?urlpath=tree`

### 3) Open a specific file/folder in JupyterLab

`https://mybinder.org/v2/gh/<owner>/<repo>/<ref>?urlpath=lab/tree/<path/in/repo>`

### 4) Build from environment repo, then pull a second repo

`https://mybinder.org/v2/gh/<env-owner>/<env-repo>/<ref>?urlpath=git-pull?repo=https://github.com/<content-owner>/<content-repo>.git`

## Direct links for this repository content

After pull, your course content is under `binder-notebooks/`.

### Lesson 2.1 - Notebook 1

- EN assignment:
  - `lesson_2_1/notebooks/notebook_1/L2_1_N1_Assignment_EN.ipynb`
- RO assignment:
  - `lesson_2_1/notebooks/notebook_1/L2_1_N1_Assignment_RO.ipynb`
- EN solved:
  - `lesson_2_1/notebooks/notebook_1/L2_1_N1_Assignment_solved_EN.ipynb`
- RO solved:
  - `lesson_2_1/notebooks/notebook_1/L2_1_N1_Assignment_solved_RO.ipynb`

### Lesson 2.1 - Notebook 2

- EN assignment:
  - `lesson_2_1/notebooks/notebook_2/L2_1_N2_Assignment_EN.ipynb`
- RO assignment:
  - `lesson_2_1/notebooks/notebook_2/L2_1_N2_Assignment_RO.ipynb`
- EN solved:
  - `lesson_2_1/notebooks/notebook_2/L2_1_N2_Assignment_solved_EN.ipynb`
- RO solved:
  - `lesson_2_1/notebooks/notebook_2/L2_1_N2_Assignment_solved_RO.ipynb`

### Theory and quiz

- Theory EN: `lesson_2_1/theory/L2_1_Theory_EN.md`
- Theory RO: `lesson_2_1/theory/L2_1_Theory_RO.md`
- Quiz EN: `lesson_2_1/quiz/L2_1_Quiz_EN.md`
- Quiz RO: `lesson_2_1/quiz/L2_1_Quiz_RO.md`

## Optional: launch directly to a notebook after pull

You can chain `git-pull` and a target path using `urlpath` (URL-encoded). Example (Notebook 1 EN):

`https://mybinder.org/v2/gh/SigmoidAI/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps://github.com/SigmoidAI/binder-notebooks.git%26urlpath%3Dlab/tree/binder-notebooks/lesson_2_1/notebooks/notebook_1/L2_1_N1_Assignment_EN.ipynb`

If this feels too complex, use the quick-start link first, then open files from the file browser.
