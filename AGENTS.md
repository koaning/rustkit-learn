# Agent Guidelines

Guidelines for AI agents working on this project.

## Notebooks

We strictly use **marimo** notebooks (`.py` files) instead of Jupyter notebooks (`.ipynb`).

- All notebooks should be created as marimo notebooks
- Use `marimo edit notebook.py` to create/edit notebooks
- Run notebooks with `uv run notebook.py`
- Always validate notebooks with `uvx marimo check <notebook.py>` before committing

## Organisation 

We do our best to mimic scikit-learn here. So just like scikit does `from sklearn.linear_model import Ridge` we should do `from rklearn.linear_model import Ridge`. We should also do our best to have objects that allow for `.get_params()`. 
