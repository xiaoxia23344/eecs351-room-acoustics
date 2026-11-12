# Group website: Room Acoustics with Impulse Responses

Static site with no build step. Open `index.html` locally, or serve it with:

    python3 -m http.server 8351   # then visit http://localhost:8351

## Pages (match the 5 progress-report points)
| Page | Covers |
| --- | --- |
| index.html | Landing page / broad overview (point 1) |
| data.html | Description of our data (point 1) |
| plots.html | Three new plots, described in detail; Plot 1 includes every data type (point 2) |
| progress.html | What we did so far, challenges, how to overcome them (point 3) |
| plan.html | Tasks for the next three weeks (point 4) |
| learned.html | One thing we learned: regularized deconvolution (point 5) |

## Publish on GitHub Pages
1. Create a public GitHub repo (e.g. `eecs351-room-acoustics`).
2. Upload everything inside this `website/` folder to the repo root (index.html must be at the top level).
3. Repo → Settings → Pages → Source: "Deploy from a branch", Branch: `main`, folder `/ (root)` → Save.
4. After about 1 minute the site is at `https://<username>.github.io/eecs351-room-acoustics/`. Submit that URL.

Total size is ~15 MB (audio + figures), well under GitHub Pages limits.
