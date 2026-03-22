# Binder Notebooks

This repository contains interactive Jupyter notebooks for different lessons, organized with support for multiple languages and learning paths.

## 📚 Project Structure

```
binder-notebooks/
├── lesson_2_1/
│   ├── notebooks/          # Interactive Jupyter notebooks
│   │   ├── notebook_1/    # Lesson 2.1 - Notebook 1
│   │   └── notebook_2/    # Lesson 2.1 - Notebook 2
│   ├── theory/            # Theoretical content (EN, RO)
│   └── quiz/              # Quiz questions (EN, RO)
└── README.md
```

## 🚀 Accessing Notebooks via Binder

### What is Binder?

Binder allows you to run Jupyter notebooks in the cloud without installing anything locally. Click a link and get an interactive notebook environment instantly!

### URL Structure

The Binder link has the following structure:

```
https://mybinder.org/v2/gh/{GITHUB_ORGANIZATION}/{BINDER_ENV_REPO}/{BRANCH}?urlpath=git-pull%3Frepo%3D{NOTEBOOKS_REPO}%26branch%3D{BRANCH}%26urlpath%3D{PATH_TO_OPEN}
```

**Components breakdown:**

| Component | Value | Description |
|-----------|-------|-------------|
| `{GITHUB_ORGANIZATION}` | `Sigmoid-Learning-Platform-Org` | GitHub organization hosting the Binder environment |
| `{BINDER_ENV_REPO}` | `binder-env` | Repository containing the Binder configuration and dependencies |
| `{BRANCH}` | `main` | Branch to use from the binder-env repository |
| `{NOTEBOOKS_REPO}` | `https://github.com/SigmoidAI/binder-notebooks.git` | This repository URL |
| `{BRANCH}` | `{branch_name}` | Branch of the notebooks repository to pull (main, develop, etc.) |
| `{PATH_TO_OPEN}` | `lab/tree/binder-notebooks` | JupyterLab path to open automatically |

### How to Generate Binder Links

Use this Python template to generate links for different branches:

```python
def create_binder_link(branch_name="main"):
    """
    Generate a Binder link for a specific branch.
    
    Args:
        branch_name (str): Branch name to access (default: "main")
    
    Returns:
        str: Full Binder URL
    """
    return f"https://mybinder.org/v2/gh/Sigmoid-Learning-Platform-Org/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps://github.com/SigmoidAI/binder-notebooks%26branch%3D{branch_name}%26urlpath%3Dlab/tree/binder-notebooks"

# Examples:
main_link = create_binder_link("main")
develop_link = create_binder_link("develop")
```

### Common Binder Links

#### Main Branch (Stable)
```
https://mybinder.org/v2/gh/Sigmoid-Learning-Platform-Org/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps://github.com/SigmoidAI/binder-notebooks%26branch%3Dmain%26urlpath%3Dlab/tree/binder-notebooks
```

#### Development Branch
```
https://mybinder.org/v2/gh/Sigmoid-Learning-Platform-Org/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps://github.com/SigmoidAI/binder-notebooks%26branch%3Ddevelop%26urlpath%3Dlab/tree/binder-notebooks
```

## 📖 Content Organization

### Lesson 2.1

#### Notebooks
- **Notebook 1** (`lesson_2_1/notebooks/notebook_1/`)
  - `L2_1_N1_Assignment_EN.ipynb` - Assignment in English
  - `helper_utils.py` - Utility functions
  - `unittests.py` - Unit tests for validation

- **Notebook 2** (`lesson_2_1/notebooks/notebook_2/`)
  - `L2_1_N2_Assignment_EN.ipynb` - Assignment in English
  - `L2_1_N2_Assignment_RO.ipynb` - Assignment in Romanian
  - `L2_1_N2_Assignment_solved_EN.ipynb` - Solved version (English)
  - `L2_1_N2_Assignment_solved_RO.ipynb` - Solved version (Romanian)
  - `helper_utils.py` - Utility functions
  - `unittests.py` - Unit tests for validation

#### Theory
- `L2_1_Theory_EN.md` - Theoretical content in English
- `L2_1_Theory_RO.md` - Theoretical content in Romanian

#### Quiz
- `L2_1_Quiz_EN.md` - Quiz questions in English
- `L2_1_Quiz_RO.md` - Quiz questions in Romanian

## 🌐 Language Support

Content is available in multiple languages:
- **EN** - English
- **RO** - Romanian

## 🔧 Running Locally

To run notebooks locally without Binder:

1. Clone the repository:
   ```bash
   git clone https://github.com/SigmoidAI/binder-notebooks.git
   cd binder-notebooks
   ```

2. Install dependencies (from `binder-env` repository):
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter:
   ```bash
   jupyter lab
   ```

4. Navigate to your desired notebook and open it.

## 📝 Notes

- Binder links pull the latest version from the specified branch automatically
- Changes to the repository are reflected in new Binder sessions
- Each Binder session is temporary and data is lost when the session ends
- For persistent work, clone the repository and run locally

