# goal
design a course on control theroy and enginnering for beginners(with essential maths knowledge on calculus and linear algebra)

# rules
- course must be orgnized around core concepts, mathematics, and engineering
- course must be accessible to beginners
- couser should be written in tex and verified with xelatex
- each foundamental concept should be explained in depth, 
- each formula should be deduced friendly for beginners, and having python scripts to do numerical exploration and plots
- contains principles of control theory and engineering, including both classical and modern control theory

# facts
- you are under wsl2 on windows, can directly call to tools with .exe extension such as xelatex.exe
- an example dmeo.py includes algorithms for pid, lqr, kf_lqr, mrac, it has numerical explaration, but no plots, and lecture on the topics in detail and depth, so this course is taking place.

# workflow snapshot (2025-10-22)
- compile the notes with `make` (xelatex backend, output in `build/course.pdf`). install TeX Live if `xelatex` is missing.
- manage Python dependencies with `.venv`; install from `requirements.txt` and run experiments via `python demo.py <experiment> [--no-plot --output path]`.
- reusable code lives in `scripts/` (dynamics, controllers, simulation, visualization, experiments) and feeds plots/data into the LaTeX chapters.
- append technical derivations to `tex/appendices/math_appendix.tex` and tooling instructions to `tex/appendices/tooling_appendix.tex` as content matures.
- generate lecture figures with `python -m scripts.tutorials <tutorial>` (tutorial options: `modeling`, `time`, `frequency`, `modern`, `adaptive`, `learning`, `digital`, `adrc`, `classical`); outputs go to `figures/` and are automatically linked in the notes.
