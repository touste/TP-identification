# Identification — Master BMED

Material parameter identification lab for the Master BMED program at Mines Saint-Étienne.

Students fit linear elastic, Neo-Hookean, and Fung constitutive models to
experimental tensile-test data (rubber, aortic tissue), then perform inverse
FEA-based identification on a plate-with-hole geometry validated against DIC
strain fields.

## Quick Start (local, recommended)

Install [pixi](https://pixi.sh), then:

```bash
pixi install
pixi run start
```

This launches JupyterLab with all dependencies (FEniCSx, PyVista, Trame,
Gmsh, etc.) pre-configured via conda-forge.

## Usage Options

| Method | Command / Link |
|--------|---------------|
| **Pixi (local)** | `pixi install && pixi run start` |
| **Binder** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.emse.fr%2Fpierrat%2Ftp-identification/master?labpath=Identification.ipynb) |
| **GitHub Codespaces** | [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/touste/TP-identification?quickstart=1) |
| **VS Code Dev Container** | [![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/touste/TP-identification) |

> **Note:** Binder builds can be slow the first time. For the best experience,
> use pixi locally or GitHub Codespaces.

## Dependencies

Managed via [pixi](https://pixi.sh) (`pixi.toml` + `pixi.lock`).
Key packages:

- [FEniCSx](https://fenicsproject.org/) (v0.11) — finite element solver
- [PyVista](https://pyvista.org/) + [Trame](https://kitware.github.io/trame/) — interactive 3D visualisation
- [Gmsh](https://gmsh.info/) — mesh generation
- [SciPy](https://scipy.org/) — optimisation (`least_squares`)
- [Matplotlib](https://matplotlib.org/) + [SciencePlots](https://github.com/garrettj403/SciencePlots) — publication-quality figures

