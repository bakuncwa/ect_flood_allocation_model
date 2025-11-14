# 💵 BantayAyuda: AI-Driven DSWD Emergency Cash Transfer System

💻 **Web App Repository:** [https://github.com/YashleyChua/BantayAyuda](https://github.com/YashleyChua/BantayAyuda)

**🏆 Overall Champion** at **Hackamare: AI Hackathon 2025** (iACADEMY Makati), themed *Innovating with LLMs*.
Developed by **TryCatchers**, Developers Society (DevSoc) Benilde.

---

## 📍 Project Description

**BantayAyuda** is a Django-based AI web application that automates **Emergency Cash Transfer (ECT)** distribution for flood-affected communities. Combining machine learning, geospatial mapping, and SMS alert integration, it enhances the speed, transparency, and accuracy of post-disaster financial assistance. Designed for **LGUs**, **DSWD personnel**, and **residents** in urban flood-prone areas, the system reduces traditional needs assessment processes from **weeks to hours**.

---

## 💡 Objectives

On **November 13, 2025**, during iACADEMY Makati’s **Hackamare: AI Hackathon 2025**, our team developed an AI-powered workflow that streamlines disaster damage assessments and automates cash assistance evaluation. BantayAyuda supports disaster-response units by improving speed, reducing human error, and ensuring equitable assistance distribution.

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

## 🔧 Tech Stack

| Category             | Technology / Library                  |
| -------------------- | ------------------------------------- |
| **Backend**          | Python, Django, Django REST Framework |
| **Frontend**         | HTML, CSS, JavaScript, Leaflet.js     |
| **Machine Learning** | CatBoost, scikit-learn, pandas        |
| **Database**         | SQLite3                               |
| **GIS Mapping**      | OpenStreetMap, Leaflet.js             |
| **Automation**       | Prototype SMS Integration             |
| **Dashboard**        | Django Admin                          |

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

* Constructed a predictive CatBoost GBDT damage-classification and Emergency Cash Transfer (ECT) classification model using engineered features such as **flood-depth:house-damage ratio** and geospatial proximity; achieved **100% accuracy, 100% precision, 100% recall**, and **1.00 F1-score**.
* Loaded the trained model into a **Django REST API** with JSON-based inference endpoints and real-time `.bin` model loading for automated ECT computation fed from sqlite3 database.
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
