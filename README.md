# 💵 BantayAyuda: AI-Driven DSWD Emergency Cash Transfer System

💻 **Web App Repository:** [https://github.com/YashleyChua/BantayAyuda](https://github.com/YashleyChua/BantayAyuda)

**🏆 Overall Champion** at **Hackamare: AI Hackathon 2025** (iACADEMY Makati), themed *Innovating with LLMs*.
Developed by **TryCatchers**, Developers Society (DevSoc) Benilde.

---

## 📍 Project Description

**BantayAyuda** is a Django-based AI web application that automates **Emergency Cash Transfer (ECT)** distribution for flood-affected communities. Combining machine learning, geospatial mapping, and SMS alert integration, it enhances the speed, transparency, and accuracy of post-disaster financial assistance. Designed for **LGUs**, **DSWD personnel**, and **residents** in urban flood-prone areas, the system reduces traditional needs assessment processes from **weeks to hours**.

---

## 💡 Objectives

On **November 13, 2025**, during iACADEMY Makati's **Hackamare: AI Hackathon 2025**, our team developed an AI-powered workflow that streamlines disaster damage assessments and automates cash assistance evaluation. BantayAyuda supports disaster-response units by improving speed, reducing human error, and ensuring equitable assistance distribution.

---

## 💡 Problem Statement

Flood-affected communities often suffer from:

* Slow, manual post-disaster needs assessments (PDNA)
* Inaccurate or inconsistent damage classification
* Delayed release of emergency cash transfers
* Lack of transparency in beneficiary selection and disbursement

Frequent typhoons in the Philippines cause widespread displacement, structural damage, and income loss. Improving data-driven relief allocation is critical for community resilience.

**Target Users:**
Local Government Units (LGUs), Department of Social Welfare and Development (DSWD) administrators, and residents eligible for ECT.

---

## 💡 Solution Pathways

BantayAyuda improves disaster-response workflows by:

* Classifying household damage levels using ML.
* Automatically determining **cash assistance amounts** (₱0 / ₱5,000 / ₱10,000).
* Mapping flood zones in real time through GIS.
* Sending simulated SMS qualification updates.
* Providing admin dashboards for LGUs/DSWD.

**Solution Features:**

| Feature Type  | Feature Description                                                                  |
| ------------- | ------------------------------------------------------------------------------------ |
| **Core**      | Damage classification (none, partial, total) via CatBoost; Automated ECT computation |
| **Enabling**  | Prototype SMS alerts; GIS hazard mapping (Leaflet.js + OpenStreetMap)                          |
| **Enhancing** | Django Admin dashboard, real-time visualization, interpretable ML metrics            |

---

## 📊 Machine Learning Model

* Built using **CatBoost Gradient Boosted Decision Trees (GBDT)**.
* Achieved **100% accuracy**, **100% precision**, **100% recall**, and **1.00 F1 score** on structured theoretical data.
* Damage categories:

  * **0** — No Damage
  * **5000** — Partial Damage
  * **10000** — Total Damage

To refine classifications, the model includes a **flood depth : house damage ratio**, comparing:

* measured / estimated **flood height (m or cm)**
* **house structural resistance** (materials, elevation, wall type)
* predefined hazard thresholds based on PDNA guidelines

This ratio improves reliability in distinguishing partial vs. total damage.

### Model Inputs Include:

* Flood depth & duration
* Type of housing material (GI sheets, concrete, light materials)
* Geolocation proximity to floodplain
* Flood:House damage ratio

Integrated directly into Django through `.bin` model loading for real-time inference.

---

## 🗂️ Project Structure

```
ect_allocation_model/
├── etl/
│   ├── scrape_waterlevel.py      # ETL: DOST-ASTI flood sensor data
│   └── scrape_households.py      # ETL: Microsoft Building Footprints
├── notebooks/
│   └── ect_allocation_model.ipynb # Main ML training & evaluation notebook
├── models/
│   └── *.bin                      # Saved CatBoost model binaries
├── data/
│   ├── waterlevel/                # Cached flood sensor CSV data
│   └── households/                # Cached building footprint GeoJSONL data
├── .gitignore
└── README.md
```

---

## 🔄 ETL Pipeline

The model is powered by two automated ETL (Extract, Transform, Load) modules that scrape real Philippine government and open-source geospatial data.

### `etl/scrape_waterlevel.py` — Flood Sensor Data

| Detail | Description |
| --- | --- |
| **Source** | [DOST-ASTI PhilSensors](https://philsensors.asti.dost.gov.ph/site/waterlevel) — real-time water level monitoring stations across the Philippines |
| **Method** | Scrapes the DOST-ASTI monitoring endpoint via session-authenticated POST request with CSRF token extraction |
| **Output** | `df_hist` (raw sensor readings per location/hour) and `df_hist_agg` (aggregated mean `flood_depth_m` and `sensor_flood_depth_m` per location) |
| **Columns** | `Location`, `Province`, `Region`, `sensor_flood_depth_m`, `flood_depth_m`, hourly readings, `date` |
| **Coverage** | ~43 flood monitoring stations nationwide |
| **Caching** | Timestamped CSV files saved to `data/waterlevel/` |
| **Fallback** | Raises `ValueError` if scraping fails — no synthetic data generation |

**Key Functions:**
- `run_dost_etl()` — Main ETL pipeline; scrapes sensor data and validates required columns
- `load_and_aggregate_flood_data()` — Loads historical data and aggregates by location
- `refresh_realtime_waterlevel()` — On-demand cache refresh with latest sensor readings

### `etl/scrape_households.py` — Building Footprint Data

| Detail | Description |
| --- | --- |
| **Source** | [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) — `philippines.geojsonl.zip` (~5 GB, 17.4M building polygons) |
| **Method** | Downloads the GeoJSONL archive, extracts it, and streams features using **Fiona** with spatial bbox filtering for efficient memory usage |
| **Output** | `household_df` with `Latitude`, `Longitude`, `House_Surface_Area_Sqm` per building |
| **Bbox Filter** | Metro Manila region: `(120.85, 14.45, 121.15, 14.85)` — yields ~922,905 building footprints |
| **Area Calculation** | Geometries projected to EPSG:3857 for accurate area computation in square meters |
| **Caching** | Raw GeoJSONL cached to `data/households/`; re-extracted from zip on first run |
| **Fallback** | Manual line-by-line GeoJSON parser if Fiona/GeoPandas unavailable |

**Key Functions:**
- `load_household_measurements()` — Main entry point; supports `bbox`, `cache_dir`, and `max_rows` parameters
- `_load_geojsonl_with_geopandas()` — Fiona-based streaming loader with spatial filtering
- `_load_geojsonl_manual()` — Fallback line-by-line parser with centroid/area calculation
- `generate_base_training_data()` — Combines flood locations with building footprints, assigns `Damage_Classification` based on flood-depth-to-height ratio

### Training Data Generation

`generate_base_training_data()` merges the two ETL outputs:

1. **Tiles** flood sensor locations cyclically to match the number of building footprints
2. **Derives** `House_Height` from `sqrt(House_Surface_Area_Sqm)` as a structural proxy
3. **Computes** `Damage_Classification` (`Not Damaged` / `Partially Damaged` / `Totally Damaged`) using flood-depth-to-height ratio thresholds (0.4 and 0.8)
4. **Produces** a training DataFrame with columns: `Location`, `Latitude`, `Longitude`, `Flood_Depth_Meters`, `House_Surface_Area_Sqm`, `Sensor_Flood_Depth_Meters`, `Damage_Classification`

Feature engineering (Spark SQL) then adds `Flood_Height_Ratio`, `Flood_House_Damage_Ratio`, and the target variable `Target_ECT_Amount`.

---

## 🔧 Tech Stack

| Category             | Technology / Library                  |
| -------------------- | ------------------------------------- |
| **Backend**          | Python, Django, Django REST Framework |
| **Frontend**         | HTML, CSS, JavaScript, Leaflet.js     |
| **Machine Learning** | CatBoost, scikit-learn, pandas, PySpark |
| **Geospatial**       | GeoPandas, Fiona, Shapely             |
| **Data Sources**     | DOST-ASTI PhilSensors, Microsoft Building Footprints |
| **Database**         | SQLite3                               |
| **GIS Mapping**      | OpenStreetMap, Leaflet.js             |
| **Automation**       | Prototype SMS Integration             |
| **Dashboard**        | Django Admin                          |
| **Visualization**    | Seaborn, Matplotlib                   |

---

## 🏆 Key Takeaways

* Automated PDNA reduces response time and improves consistency.
* GIS layers help LGUs visualize hazard zones immediately.
* ML-driven classification ensures fair and transparent ECT distribution.
* Scalable architecture can be deployed nationwide for future typhoon seasons.

---

## 📊 System Demo 

<p align="center">
  <img src="https://github.com/bakuncwa/ect_flood_allocation_model/blob/main/bantayayuda_1.png?raw=true" 
       alt="Bantay Ayuda System Demo" width="800"/>
</p>
<p align="center">
  <img src="https://github.com/bakuncwa/ect_flood_allocation_model/blob/main/bantayayuda_2.png?raw=true" 
       alt="Bantay Ayuda System Demo" width="800"/>
</p>
<p align="center">
  <img src="https://github.com/bakuncwa/ect_flood_allocation_model/blob/main/bantayayuda_3.png?raw=true" 
       alt="Bantay Ayuda System Demo" width="800"/>
</p>

---

## 👥 Team TryCatchers – Roles and Responsibilities

| Name                         | Role                                                       |
| ---------------------------- | ---------------------------------------------------------- |
| **Gabrielle Ysabel Almirol** | Machine Learning Developer, Front-End & Back-End Developer |
| **Yashley Joaquin Chua**     | Researcher, Front-End & Back-End Developer                 |
| **Xander Julius Abo**        | Researcher, Front-End Developer                            |

### Technical Contributions

* Built an automated **ETL pipeline** web-scraping real-time flood sensor data from **DOST-ASTI PhilSensors** (43 stations) and streaming **922,905 Metro Manila building footprints** from **Microsoft Global ML Building Footprints** (17.4M features, ~5 GB GeoJSONL) using **Fiona** with spatial bbox filtering for memory-efficient geospatial ingestion.
* Engineered features via **PySpark / Spark SQL** — computed `Flood_Height_Ratio` and `Flood_House_Damage_Ratio` (sensor-confirmed) from building surface area projections (**EPSG:3857**) and flood depth readings; derived `Damage_Classification` using ratio thresholds (0.4 / 0.8).
* Constructed a predictive **CatBoost GBDT** multi-class classifier (iterations=1000, depth=6, lr=0.1, MultiClass loss, TotalF1 eval metric) for ECT allocation (₱0 / ₱5,000 / ₱10,000); achieved **99.99% accuracy**, **1.00 weighted F1-score**, **1.00 precision**, and **1.00 recall** on 922K real building footprints.
* Generated a **Seaborn correlation heatmap** (YlGn colormap, zero-variance column filtering) to visualize feature-target relationships — confirming strong correlations between `Flood_House_Damage_Ratio`, `Flood_Height_Ratio`, and `Target_ECT_Amount`, validating the engineered ratio features as primary drivers of ECT classification.
* Loaded the trained `.bin` model into a **Django REST API** with JSON-based inference endpoints and real-time model loading for automated ECT computation fed from SQLite3 database.
* Implemented **Leaflet.js + OpenStreetMap GIS mapping** with dynamic JavaScript logic, hazard visualization layers, and optimized Django database interactions.

---

## 📌 References

1. **DSWD (2025). Emergency Cash Transfer (ECT) Program Guidelines.**
   [https://www.dswd.gov.ph](https://www.dswd.gov.ph)
2. **Philippine Statistics Authority (2025). NICTHS 2024 Survey Findings.**
   [https://psa.gov.ph/nichths-2024](https://psa.gov.ph/nichths-2024)
3. **NDRRMC (2025). SitRep No. 5: Typhoon Uwan (Fung-wong).**
   [https://ndrrmc.gov.ph](https://ndrrmc.gov.ph)
4. **NDRRMC (2025). SitRep No. 3: Typhoon Tino (Kalmaegi).**
   [https://ndrrmc.gov.ph](https://ndrrmc.gov.ph)
5. **DBM (2025). National Calamity Fund Utilization Report.**
   [https://www.dbm.gov.ph/calamity-fund-2025](https://www.dbm.gov.ph/calamity-fund-2025)
6. **Office of Civil Defense (2025). PDNA Preliminary Report: Typhoon Uwan.**
   [https://ocd.gov.ph/pdna-uwan-2025](https://ocd.gov.ph/pdna-uwan-2025)
7. **PAGASA (2025). Typhoon Season Summary.**
   [https://www.pagasa.dost.gov.ph/typhoon-report-2025](https://www.pagasa.dost.gov.ph/typhoon-report-2025)
8. **Republic Act No. 10639. Free Mobile Disaster Alerts Act.**
   [https://www.officialgazette.gov.ph/2014/06/20/republic-act-no-10639/](https://www.officialgazette.gov.ph/2014/06/20/republic-act-no-10639/)
9. **WHO Philippines (2025). SPEED System Performance Report.**
   [https://www.who.int/philippines/speed-2025](https://www.who.int/philippines/speed-2025)
10. **Smart Communications (2025). Emergency Alert System Report.**
    *(Internal Report — No public link)*
11. **Globe Telecom (2025). Disaster Response SMS Gateway Specifications.**
    [https://www.globe.com.ph/disaster-response](https://www.globe.com.ph/disaster-response)
12. **Litslink (2025). Cost Estimation for GIS-Based Disaster Aid Platforms.**
    [https://litslink.com/cost-estimate-gis-app](https://litslink.com/cost-estimate-gis-app)
13. **DOST-ASTI PhilSensors — Real-Time Water Level Monitoring** *(used by `scrape_waterlevel.py`)*
    [https://philsensors.asti.dost.gov.ph/site/waterlevel](https://philsensors.asti.dost.gov.ph/site/waterlevel)
14. **Microsoft Global ML Building Footprints — Philippines GeoJSONL** *(used by `scrape_households.py`)*
    [https://github.com/microsoft/GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints)
    Direct download: [https://minedbuildings.z5.web.core.windows.net/legacy/southeast-asia/philippines.geojsonl.zip](https://minedbuildings.z5.web.core.windows.net/legacy/southeast-asia/philippines.geojsonl.zip)
