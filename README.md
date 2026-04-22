# 🛡️ Digital Shield — Ticket Anomaly Detection System

> **Version:** 1.0.0 | **Python:** 3.11+ | **Database:** Supabase PostgreSQL

A production-style, modular Python backend system that connects to a Railway Booking PostgreSQL database, detects suspicious ticket booking patterns using a hybrid **rule-based + ML engine**, and generates actionable reports.

---

## 📌 Project Overview

The Digital Shield system scans railway ticket records from the `tickets_1` table and identifies anomalous booking behaviour using:

- **8 rule-based anomaly detectors** (deterministic, interpretable)
- **Isolation Forest ML model** (statistical, unsupervised)
- **Risk scoring engine** (0–100+ numeric score → LOW / MEDIUM / HIGH)

It is designed to handle **900,000+ rows** efficiently using server-side PostgreSQL cursors and chunked pandas operations.

---

## 🚨 Anomaly Types Detected

| # | Rule | Score Weight | Description |
|---|------|-------------|-------------|
| 1 | Adult + Many Minors | **35** | 1 adult travelling with 3+ minors on same PNR |
| 2 | Bulk Booking | **20** | Same user makes >5 bookings in one day |
| 3 | Same IP Bookings | **20** | Same IP used for >5 bookings in one day |
| 4 | Last-Minute Booking | **15** | Journey booked same day as travel |
| 5 | Rapid Transactions | **15** | Multiple bookings by same user within 10 minutes |
| 6 | Repeated Same Route | **10** | Same user books identical route >3 times |
| 7 | Large Group | **20** | Single PNR has >6 passenger entries |
| 8 | Bank High Frequency | **10** | Same bank used for >20 transactions/day |

**Risk Levels:**

| Score Range | Level |
|-------------|-------|
| 0 – 29 | 🟢 LOW |
| 30 – 59 | 🟡 MEDIUM |
| 60+ | 🔴 HIGH |

---

## 🗂️ Project Structure

```
project_root/
├── app.py                  # Main pipeline runner
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env.example            # Environment variable template
├── outputs/                # Generated report files
├── logs/                   # Application log files
└── src/
    ├── db.py               # PostgreSQL connection (psycopg2)
    ├── loader.py           # Data loading from tickets_1
    ├── preprocess.py       # Data cleaning & feature engineering
    ├── anomalies.py        # 8 rule-based anomaly detectors
    ├── scoring.py          # Risk score calculation
    ├── model.py            # Isolation Forest ML detection
    ├── reports.py          # CSV / JSON / summary exports
    └── utils.py            # Logger setup & shared helpers
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites

- Python **3.11 or higher**
- Access to a Supabase PostgreSQL database with the `tickets_1` table

---

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Copy the example env file and fill in your credentials:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then open `.env` and replace `your_password_here` with your actual Supabase password:

```env
DB_HOST=aws-1-ap-south-1.pooler.supabase.com
DB_PORT=6543
DB_NAME=postgres
DB_USER=postgres.trssafmvdbeagsdnivcl
DB_PASSWORD=your_actual_password
```

> ⚠️ **Never commit your `.env` file to Git.** Add it to `.gitignore`.

---

### 5. Run the System

```bash
python app.py
```

The pipeline will:
1. Connect to the database
2. Load all rows from `tickets_1`
3. Clean and preprocess the data
4. Run 8 rule-based anomaly detectors
5. Run Isolation Forest ML detection
6. Calculate risk scores
7. Export reports to `outputs/`
8. Print a summary to the console

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `outputs/suspicious_bookings.csv` | All flagged rows with scores and triggered rules |
| `outputs/suspicious_bookings.json` | Same data in JSON format (for API consumption) |
| `outputs/dashboard_summary.csv` | Aggregated risk level counts and percentages |
| `outputs/top_risky_users.csv` | Top 5 users by cumulative risk score |
| `outputs/top_risky_routes.csv` | Top 5 routes by cumulative risk score |
| `logs/app.log` | Full application log with timestamps |

---

## 🔬 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2.2 | Data loading and transformation |
| `numpy` | 1.26.4 | Numerical operations |
| `psycopg2-binary` | 2.9.9 | PostgreSQL connection |
| `scikit-learn` | 1.5.0 | Isolation Forest ML model |
| `python-dotenv` | 1.0.1 | .env credential loading |

---

## 🔮 Future Roadmap (V2+)

The codebase includes ready-to-activate hooks for:

| Feature | Hook Location | Status |
|---------|--------------|--------|
| **CCTV Integration** | `utils.cctv_integration_hook()` | 🔜 Planned |
| **Email Alerts** | `utils.email_alert_hook()` | 🔜 Planned |
| **Dashboard API** | `utils.dashboard_api_hook()` | 🔜 Planned |

---

## 📊 Database Table Reference

**Table:** `tickets_1`

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `psgn_name` | Passenger name |
| `train_number` | Train number |
| `cls` | Class (SL, 3A, 2A, 1A) |
| `txn_date` | Transaction date |
| `txn_time` | Transaction time |
| `ip_addrs` | Booking IP address |
| `jrny_date` | Journey date |
| `pnrno` | PNR number (passenger group) |
| `from_stn` | Origin station |
| `to_stn` | Destination station |
| `age` | Passenger age |
| `sex` | Passenger gender |
| `quota` | Booking quota |
| `coach_no_seat_no` | Coach and seat assignment |
| `txntype` | Transaction type |
| `bank_name` | Bank used for payment |
| `txn_no` | Transaction number |

---

## 👨‍💻 Developer Notes

- **Memory safety:** Uses named server-side psycopg2 cursors + 50,000-row chunks
- **No credentials in code:** All secrets loaded via `.env` only
- **Beginner-friendly:** Each module has a single clear responsibility
- **Extendable:** Add new rule flags in `anomalies.py` and scoring weights in `scoring.py`

---

*Digital Shield Ticket Anomaly Detection System — Version 1.0*
