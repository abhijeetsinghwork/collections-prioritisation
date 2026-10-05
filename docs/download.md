# Getting the data

This project uses the **Freddie Mac Single-Family Loan-Level Dataset (SFLLD)**.
It is free, but it requires registration and it is **not redistributed in this
repository** — Freddie Mac's terms permit publishing code and analytical results,
not the raw data. You need to download it yourself. It takes about 15 minutes.

---

## 1. Register

Go to the dataset page:

> https://freddiemac.com/research/datasets/sf-loanlevel-dataset

Click through to **Clarity Data Intelligence**, the download portal:

> https://claritydownload.fmapps.freddiemac.com/CRT/

Create an account. Registration is free. You will have to accept the Single-Family
Loan-Level Dataset terms and conditions, which permit use for personal, internal,
academic and research purposes.

---

## 2. Go to the SFLLD Data Download page

Inside Clarity, navigate to the **SFLLD** section. You will see several options:

| Option | Use it? |
|---|---|
| Full dataset | **No.** ~56 million mortgages. Far too large for a laptop. |
| Standard Dataset by year | No. Still very large per year. |
| Non-Standard Dataset | No. One-off 2021 publication of excluded loans. |
| **Sample Dataset** | **Yes. This is what you want.** |
| RPL Mapping file | No. |

The **sample dataset** is a simple random sample of 50,000 loans from each full
vintage year, with their complete monthly performance history. It is the right
size for local work and still gives a genuine monthly delinquency panel.

---

## 3. Download these vintages

Download the sample file for each of these years:

```
2000 2001 2002 2003 2004 2005 2006 2007 2008 2009
2010 2011 2012 2013 2014 2015 2016 2017 2018 2019
2020 2021
2022 2023 2024
```

**Why 2020 and 2021 are included even though they are excluded from training:**
mass pandemic forbearance changes what "delinquent" means in the data. Those two
years are held out as a real, dated concept-drift event for the monitoring stage.
Download them, but do not train on them.

If you want to start smaller, download **2015 alone** first, get the pipeline
working end to end on it, then come back for the rest. This is the recommended
path.

---

## 4. Also download the documentation

From the same page, grab:

- **File Layout** (Excel) — the authoritative column list and ordering
- **General User Guide** (PDF) — field definitions, delinquency codes,
  zero-balance codes
- **File Headers** (zip), if offered

These matter. The layout has been revised across releases, so the pipeline's
schema is generated from the layout file rather than hardcoded. Put them in
`data/raw/docs/`.

---

## 5. Unzip into place

Each year downloads as `sample_YYYY.zip` containing two files:

```
sample_orig_YYYY.txt    one row per loan, static origination attributes
sample_perf_YYYY.txt    one row per loan-month, performance history
```

Both are **pipe-delimited** with **no header row**.

Unzip everything into `data/raw/` so it looks like this:

```
data/raw/
├── docs/
│   ├── file_layout.xlsx
│   └── general_user_guide.pdf
├── sample_orig_2000.txt
├── sample_perf_2000.txt
├── sample_orig_2001.txt
├── sample_perf_2001.txt
└── ...
```

On macOS:

```bash
cd data/raw
for f in sample_*.zip; do unzip -o "$f"; done
rm sample_*.zip        # optional, once you have confirmed the .txt files exist
```

---

## 6. Verify before building

```bash
# Count the files — should be 2 per vintage year
ls data/raw/sample_*.txt | wc -l

# Peek at the first row of an origination file
head -1 data/raw/sample_orig_2015.txt

# Confirm it is pipe-delimited and has no header
head -1 data/raw/sample_perf_2015.txt
```

The first row should be data, not column names. If you see column names,
you have downloaded a variant with headers and the ingest schema needs the
header row skipped.

**Critically — confirm git is ignoring the data before you commit anything:**

```bash
git check-ignore -v data/raw/sample_orig_2015.txt
```

This must print a matching `.gitignore` rule. If it prints nothing, stop and fix
`.gitignore` before committing. Purging a large data file from git history after
the fact is painful.

---

## 7. Macro data (optional, for the macro feature family)

State-level unemployment and house price indices come from FRED, which is free.

Either register for an API key at https://fred.stlouisfed.org/docs/api/api_key.html
and let the pipeline fetch them, or download these series as CSV manually:

| Series | What it is |
|---|---|
| `{STATE}UR` | State unemployment rate, e.g. `CAUR` for California |
| `{STATE}STHPI` | State house price index, e.g. `CASTHPI` |
| `MORTGAGE30US` | National 30-year fixed mortgage rate |

Put manual CSVs in `data/raw/macro/`. These are small, freely redistributable,
and may be committed if you want the repo fully reproducible.

---

## Licensing note

Freddie Mac permits use of the SFLLD for personal, internal, academic and
research purposes, and permits publishing academic or research results derived
from it. It does **not** permit redistribution of the dataset itself, and
commercial redistribution requires a licensing agreement.

Practically, for this repo: publish the code, the figures and the aggregate
results. Never commit the raw files or any derived file that reproduces loan-level
records.

---

## Sources

- [Freddie Mac Single-Family Loan-Level Dataset](https://freddiemac.com/research/datasets/sf-loanlevel-dataset)
- [Clarity Data Intelligence download portal](https://claritydownload.fmapps.freddiemac.com/CRT/)
- [SFLLD General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf)
- [SFLLD Terms and Conditions](https://capitalmarkets.freddiemac.com/crt/docs/pdfs/fre_terms_conditions_sflld.pdf)
