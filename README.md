# UV-Vis Deconvolution

A browser-based tool for resolving concentration profiles of multiple chemical species from time-resolved UV-Vis spectra. Implements both Classical Least Squares (CLS) when reference spectra are known and Multivariate Curve Resolution (MCR-ALS) when only mixture data is available.

All computation runs locally in the browser — no data is uploaded anywhere.

## Live demo

👉 [Open the app](https://YOUR-USERNAME.github.io/uv-deconvolution/)

*(Replace `YOUR-USERNAME` after deployment)*

## Features

- Synthetic 4-species kinetic demo (A→B→C→D consecutive first-order reaction)
- Upload your own mixture data via CSV
- Optional reference spectra upload for CLS analysis
- MCR-ALS with non-negativity constraints when references are unknown
- Export resolved concentration profiles as CSV

## CSV format

**Mixture data** — header row of times, first column of wavelengths:

```
wavelength,0,2,4,6,...
250,0.123,0.234,0.345,...
255,0.234,0.345,0.456,...
...
```

**Reference spectra (optional)** — header row of species names, first column of wavelengths:

```
wavelength,A,B,C,D
250,0.95,0.10,0.02,0.01
255,0.92,0.15,0.03,0.01
...
```

## Local development

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

## Build for production

```bash
npm run build
```

Output goes to `dist/`.

## Deployment

This repository auto-deploys to GitHub Pages on every push to `main` via GitHub Actions (see `.github/workflows/deploy.yml`).

## Methods

- **CLS** — solves Beer-Lambert at each timepoint via non-negative least squares.
- **MCR-ALS** — alternates between estimating concentrations and pure spectra under non-negativity constraints.

For larger datasets or hard kinetic modeling, see [pyMCR](https://github.com/usnistgov/pyMCR) or the MATLAB MCR-ALS GUI by Tauler et al.

## License

MIT
