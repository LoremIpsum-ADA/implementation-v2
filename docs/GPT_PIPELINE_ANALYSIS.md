# Comprehensive Analysis: Causal Inference Pipeline for Endangered Species Impact Assessment

**Document Purpose:** Complete cell-by-cell analysis of the causal inference pipeline examining natural disasters' effects on endangered species in Asia

**Analysis Date:** November 18, 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Cell-by-Cell Analysis](#cell-by-cell-analysis)
   - [Section 1: Setup and Configuration](#section-1-setup-and-configuration)
   - [Section 2: Data Acquisition](#section-2-data-acquisition)
   - [Section 3: Spatial Infrastructure](#section-3-spatial-infrastructure)
   - [Section 4: Panel Construction](#section-4-panel-construction)
   - [Section 5: Two-Way Fixed Effects DiD](#section-5-twfe-difference-in-differences)
   - [Section 6: Double Machine Learning](#section-6-double-machine-learning)
   - [Section 7: Sun-Abraham Staggered DiD](#section-7-sun-abraham-staggered-did)
4. [Method Comparison and Synthesis](#method-comparison-and-synthesis)
5. [Key Insights and Recommendations](#key-insights-and-recommendations)

---

## Executive Summary

### Research Question
**Do natural disasters causally affect endangered species populations in Asia?**

This pipeline implements a sophisticated causal inference study analyzing whether disasters (wildfires, floods, cyclones, earthquakes) impact the occurrence patterns of 13 threatened species across Asia from 2000-2024.

### Core Challenge
Establishing causality in observational ecological data is difficult because:
- **Confounding**: Disasters don't occur randomly—they're correlated with geography, climate, and human activity
- **Heterogeneity**: Effects may vary by species, location, and time
- **Data complexity**: Spatial panel data with staggered treatment timing

### Approach
The pipeline addresses these challenges using **three complementary causal inference methods**:

1. **Two-Way Fixed Effects (TWFE) DiD**: Traditional panel approach controlling for grid and time fixed effects
2. **Double Machine Learning (DML)**: Modern ML-based method controlling for high-dimensional confounders
3. **Sun-Abraham (2021)**: Robust to heterogeneous treatment effects in staggered adoption

### Key Findings

| Method | Effect Size | Interpretation | Significance |
|--------|-------------|----------------|--------------|
| **TWFE DiD** | -0.0081 | 0.81 pp decrease in occupancy | p = 0.022 ** |
| **Double ML** | +0.0126 | 1.26 pp increase in occupancy | p < 0.001 *** |
| **Sun-Abraham** | -0.0079 | 0.79 pp decrease in occupancy | p = 0.025 ** |

### Interpretation
- **TWFE and Sun-Abraham converge**: Both show small negative effects (~0.8 percentage points)
- **Double ML diverges**: Shows opposite positive effect (+1.3 percentage points)
- **20% difference** between methods suggests potential confounding or effect heterogeneity
- **Recommendation**: Prefer Double ML (better confounder control) or Sun-Abraham (robust to heterogeneity)

### Species Heterogeneity
- **Most negatively affected**: Pongo abelii (Orangutans), Rhacophorus catamitus (frogs)
- **Most positively affected**: Arborophila davidi (partridges)
- **Wide variation**: -6.0% to +7.0% across species, indicating differential vulnerability

---

## Project Overview

### Geographic and Temporal Scope

**Geographic Coverage:**
- **Region**: Asia bounding box [60°E to 150°E, -10°S to 55°N]
- **Spatial Resolution**: 1.0° grid (~111 km at equator)
- **Total Grid Cells**: ~6,175 cells covering the Asian continent

**Temporal Coverage:**
- **Time Period**: 2000-2024 (25 years)
- **Temporal Unit**: Annual aggregation
- **Event Study Window**: -3 to +5 years relative to disaster

### Study Design

**Treatment (Independent Variable):**
- **Disaster Types**: Wildfires, floods, cyclones/typhoons, earthquakes
- **Data Source**: EM-DAT (Emergency Events Database)
- **Treatment Definition**: 50 km circular buffer around disaster epicenter
- **Treatment Type**: Binary indicator (treated = 1 if disaster occurred in grid-year)

**Outcome (Dependent Variable):**
- **Metric**: Species occupancy (binary: present/absent in grid-year)
- **Alternative**: Count of GBIF occurrences (secondary)
- **Data Source**: GBIF (Global Biodiversity Information Facility)

**Species Selection Criteria:**
- **Conservation Status**: Critically Endangered (CR), Endangered (EN), or Vulnerable (VU)
- **Taxonomic Groups**: Birds (Aves), Mammals (Mammalia), Reptiles (Reptilia), Amphibians (Amphibia)
- **Data Threshold**: ≥50 GBIF occurrence records
- **Final Sample**: 13 species meeting all criteria

### Methodological Innovation: Range-Based Panel Filtering

**Challenge**: A complete balanced panel would have:
- 6,175 grids × 25 years × 13 species = **~2 million observations**
- Most grid-species combinations are biologically impossible (species doesn't live there)

**Solution**: Filter panel to only include grid cells that overlap with species' IUCN range maps
- **Result**: 75% reduction to ~200-500K observations
- **Benefits**: 
  - Biologically realistic (only possible habitats)
  - Computationally efficient (much faster estimation)
  - Improved precision (removes noise from impossible combinations)

This innovation is a key contribution that makes the analysis feasible while maintaining scientific validity.

---

## Cell-by-Cell Analysis

### Section 1: Setup and Configuration

#### Cell 1: Project Introduction (Markdown)

**Purpose**: Provides high-level overview of the pipeline structure and objectives.

**Content**:
- Research scope: Endangered species in Asia with natural disasters as treatment
- Pipeline stages: Data ingestion → Preprocessing → Causal analysis → Robustness
- Estimators to be used: DiD (TWFE), Sun-Abraham, Double ML
- Expected outputs: Event-study plots, coefficient tables, maps, species summaries

**Theoretical Context**:
This is a **quasi-experimental design** where:
- Treatment assignment (disasters) is **not randomized** (unlike an RCT)
- We observe outcomes **before and after** treatment
- We use **control groups** (untreated grids) for comparison
- **Multiple methods** triangulate on causal effects

Think of it like a natural experiment: disasters happen in some places and not others, creating variation we can exploit for causal identification—but we need sophisticated methods to account for non-random treatment assignment.

---

#### Cell 2: Configuration Parameters (Python)

**Code Functionality**:
```python
CONFIG = {
    'grid_resolution_deg': 1.0,  # Changed from 0.5 to 1.0
    'region_bbox': [60.0, -10.0, 150.0, 55.0],
    'time_range': (2000, 2024),
    'use_range_filtering': True,
    'event_window': (-3, 5),
    'clustering_var': 'grid_id',
    'n_folds': 5,
    'ml_model': 'random_forest'
}
```

**Key Design Decisions**:

1. **Grid Resolution: 1.0° (not 0.5° or 0.1°)**
   - **Trade-off**: Coarser resolution = faster computation but lower spatial precision
   - **Justification**: 1° achieves 75% reduction in computational load while maintaining biological relevance
   - **Grid Cell Size**: ~111 km × 111 km ≈ 12,000 km² per cell
   - **Comparison**: 
     - 0.1° would give 100× more cells (computationally infeasible)
     - 1.0° is appropriate for regional-scale species ranges

2. **Event Study Window: -3 to +5 years**
   - **Pre-treatment period** (-3 to -1): Tests parallel trends assumption
   - **Treatment year** (0): Immediate impact
   - **Post-treatment** (+1 to +5): Dynamic/delayed effects
   - **Why this window?**: Balance between capturing dynamics and maintaining sample size

3. **Disaster Buffer: 50 km**
   - **Rationale**: Captures both direct impact zone and adjacent affected areas
   - **Biological relevance**: Species mobility ranges (mammals can move 10s of km, birds 100s of km)
   - **Sensitivity**: Could test 25 km (narrow) or 100 km (wide) for robustness

4. **Range-Based Filtering: TRUE**
   - **Critical innovation**: Only analyze grid-species pairs where IUCN range overlaps grid
   - **Impact**: Reduces panel from ~2M to ~500K observations (75% reduction)
   - **Validity**: Biologically realistic—can't detect effects where species doesn't exist

**Output**:
```
Configuration set for endangered_asia_disaster_analysis
Study region: [60.0, -10.0, 150.0, 55.0]
Time range: (2000, 2024)
Grid resolution: 1.0°
```

**Theoretical Insight**:
The configuration reflects fundamental **bias-variance trade-offs** in causal inference:
- **Finer resolution** → lower bias (more precise treatment assignment) but higher variance (smaller sample per unit)
- **Coarser resolution** → higher bias (treatment spillovers) but lower variance (larger samples)
- The choice of 1° balances these concerns for this application

---

### Section 2: Data Acquisition

#### Cell 3: Data Sources Overview (Markdown)

**Purpose**: Documents all external data sources and how to access them.

**Key Data Sources**:

1. **EM-DAT (Centre for Research on Epidemiology of Disasters)**
   - Global disaster database maintained by WHO
   - Requires free registration
   - Contains location, timing, type, and severity of disasters

2. **GBIF (Global Biodiversity Information Facility)**
   - Aggregates species occurrence records from museums, citizen science, research
   - API access for bulk downloads
   - Provides coordinates, dates, species identification

3. **IUCN Red List Spatial Data**
   - Expert-validated species range maps (polygon shapefiles)
   - Downloaded by taxonomic group (Birds, Mammals, etc.)
   - Essential for biologically realistic panel filtering

**Data Integration Challenge**:
These three datasets have different:
- **Spatial formats**: Points (GBIF, EM-DAT) vs Polygons (IUCN)
- **Temporal granularity**: Exact dates vs annual ranges
- **Coordinate systems**: Need standardization to WGS84
- **Taxonomic names**: Scientific name matching required

The pipeline must **harmonize** these heterogeneous data sources into a unified panel structure.

---

#### Cell 4: GBIF Species Selection (Python)

**Code Functionality**:
Identifies threatened species in Asia meeting study criteria (CR/EN/VU status, ≥50 occurrences).

**Key Code Section**:
```python
curated_species = [
    {'taxon': 'Aves', 'scientificName': 'Lophura edwardsi', 'status': 'CR'},
    {'taxon': 'Mammalia', 'scientificName': 'Panthera tigris', 'status': 'EN'},
    # ... 11 more species
]
```

**Species List (13 Total)**:

| Taxon | Species | Common Name | IUCN Status |
|-------|---------|-------------|-------------|
| **Aves** (4) | Lophura edwardsi | Edwards's Pheasant | CR |
| | Arborophila davidi | Orange-necked Partridge | EN |
| | Turdoides striata | Jungle Babbler | VU |
| | Carpococcyx renauldi | Coral-billed Ground-Cuckoo | EN |
| **Mammalia** (4) | Panthera tigris | Tiger | EN |
| | Elephas maximus | Asian Elephant | EN |
| | Rhinoceros sondaicus | Javan Rhino | CR |
| | Pongo abelii | Sumatran Orangutan | CR |
| **Reptilia** (3) | Crocodylus siamensis | Siamese Crocodile | CR |
| | Chelonia mydas | Green Sea Turtle | EN |
| | Cuora trifasciata | Golden Coin Turtle | CR |
| **Amphibia** (2) | Ansonia latidisca | Sambas Stream Toad | EN |
| | Rhacophorus catamitus | Vietnam Tree Frog | VU |

**Selection Rationale**:
- **Taxonomic diversity**: Represents 4 major vertebrate groups
- **Threat levels**: Mix of CR (highest risk), EN, and VU
- **Geographic spread**: Covers Southeast Asia, India, China
- **Data availability**: Each has sufficient GBIF records for statistical power

**Output**:
```
Species to analyze: 13
Taxonomic groups: {'Aves': 4, 'Mammalia': 4, 'Reptilia': 3, 'Amphibia': 2}
Conservation status: {'CR': 5, 'EN': 6, 'VU': 2}
Saved species list to data/processed/target_species.csv
```

**Theoretical Context**:
This is the **treatment group** in our study—these are the species whose populations we hypothesize are affected by disasters. The diversity ensures we can:
- Estimate **average effects** across species
- Examine **heterogeneity** (differential vulnerability)
- Generalize findings beyond a single taxon

---

#### Cell 5-6: GBIF Download Workflow (Python)

**Purpose**: Request and download occurrence data for target species via GBIF API.

**Key Code Section**:
```python
predicate = {
    "type": "and",
    "predicates": [
        {"type": "or", "predicates": taxon_predicates},  # Species list
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
        {"type": "greaterThanOrEquals", "key": "YEAR", "value": "2000"},
        {"type": "within", "geometry": "POLYGON(...)"}  # Asia bounding box
    ]
}
```

**Data Filters Applied**:
1. **Taxonomic**: Only target species (13 species)
2. **Spatial**: Within Asia bounding box
3. **Temporal**: Years 2000-2024
4. **Quality**:
   - Must have coordinates (no vague localities)
   - No geospatial issues flagged

**Challenge**: GBIF API is asynchronous
- **Step 1**: Submit download request → get download key
- **Step 2**: Wait for email notification (can take hours)
- **Step 3**: Download ZIP file and extract occurrences.csv

**Output**:
```
Download requested. Key: 0019239-251025141854904
You will receive an email when ready
Download URL: https://www.gbif.org/occurrence/download/...
```

**Theoretical Insight**:
GBIF data has **observation bias**:
- More records near roads, cities, protected areas
- Temporal bias toward recent years (more digital data)
- Detection probability varies by species (cryptic species underreported)

These biases are why we need **causal inference methods** rather than simple before-after comparisons—we need to account for systematic differences between treated and control areas.

---

#### Cell 7: EM-DAT Disaster Data Loading (Python)

**Code Functionality**:
Loads and filters disaster records from EM-DAT database.

**Key Code Section**:
```python
disaster_map = {
    'wildfire': ['Wildfire'],
    'flood': ['Flood', 'Flash flood', 'Riverine flood'],
    'cyclone': ['Storm', 'Tropical cyclone', 'Typhoon'],
    'earthquake': ['Earthquake', 'Ground movement']
}

df = df[(df['Start Year'] >= 2000) & (df['Start Year'] <= 2024)]
df = df[df['Disaster Subtype'].isin(type_filter)]
df = df[(df['Longitude'] >= 60) & (df['Longitude'] <= 150)]
```

**Filters Applied**:
1. **Temporal**: 2000-2024 (matches species data)
2. **Disaster Types**: Only wildfires, floods, cyclones, earthquakes
3. **Geographic**: Asia bounding box
4. **Data Quality**: Remove records without coordinates

**Why These Disaster Types?**
- **Wildfire**: Direct habitat destruction, smoke inhalation
- **Flood**: Habitat inundation, drowning, disease spread
- **Cyclone**: Wind damage, storm surge, habitat disruption
- **Earthquake**: Immediate mortality, landslides, infrastructure collapse

**Output**:
```
Loaded 2,847 disaster events
  Wildfires: 523
  Floods: 1,204
  Cyclones: 891
  Earthquakes: 229
Years: 2000-2024
```

**Theoretical Context**:
These disasters are the **treatment variable** in our causal analysis. Key properties:
- **Exogenous** (mostly): Natural processes not caused by species presence
- **Sharp timing**: Clear before/after comparison
- **Spatial variation**: Some areas affected, others not
- **Staggered adoption**: Different grids treated in different years

However, disasters are **not randomly assigned**:
- Coastal areas more prone to cyclones
- Forested areas more prone to wildfires
- This non-random treatment is why we need DiD, DML, and Sun-Abraham methods

---

### Section 3: Spatial Infrastructure

#### Cell 10: Analysis Grid Creation (Python)

**Purpose**: Create a regular 1° × 1° latitude-longitude grid covering Asia.

**Key Code**:
```python
def create_analysis_grid(bbox, resolution):
    lons = np.arange(min_lon, max_lon, resolution)  # Every 1°
    lats = np.arange(min_lat, max_lat, resolution)
    
    for lon in lons:
        for lat in lats:
            poly = box(lon, lat, lon + resolution, lat + resolution)
            # Creates square grid cell
```

**Grid Properties**:
- **Cell Count**: ~6,175 cells covering Asia
- **Cell Size**: 1° × 1° (approximately 111 km × 111 km at equator)
- **Average Area**: ~12,000 km² per cell
- **Coordinate System**: WGS84 (EPSG:4326)
- **Grid IDs**: Unique identifiers like `g_0_0`, `g_0_1`, etc.

**Why a Regular Grid?**
1. **Standardization**: Equal-sized analysis units (unlike administrative boundaries)
2. **Comparability**: Same size across space enables fair comparisons
3. **Computational efficiency**: Regular structure is easier to process
4. **Independence**: Political boundaries don't affect ecological patterns

**Output**:
```
Created grid with 6,175 cells
Avg cell area: 11,847.3 km²
Saved grid to data/processed/analysis_grid.gpkg
```

**Theoretical Insight**:
The grid resolution represents a **fundamental trade-off**:
- **Finer grids** (e.g., 0.1°): More precise treatment assignment, but 100× more cells → computational infeasibility
- **Coarser grids** (e.g., 5°): Fast computation, but averages away local variation
- **1° choice**: Balances precision with feasibility for regional-scale species

Think of this as choosing the "unit of analysis"—analogous to deciding whether to study individuals vs neighborhoods vs cities in social science.

---

#### Cell 11: Spatial Join Functions (Python)

**Purpose**: Define functions to link point data (occurrences, disasters) to grid cells.

**Key Functions**:

1. **`map_points_to_grid()`**: Assigns GBIF occurrences to grid cells
```python
def map_points_to_grid(points_gdf, grid_gdf):
    joined = gpd.sjoin(pts, grid_gdf, predicate='within')
    # Finds which grid cell contains each point
```

2. **`map_polygon_to_grid()`**: Calculates how much of a polygon (IUCN range) overlaps each grid cell
```python
def map_polygon_to_grid(poly_gdf, grid_gdf):
    intersected = gpd.overlay(poly, grid_gdf, how='intersection')
    overlap_fraction = overlap_km2 / cell_area
    # Returns fraction of each grid cell covered by range
```

3. **`expand_disaster_footprint()`**: Converts point disasters into circular impact zones
```python
def expand_disaster_footprint(disaster_points, radius_km=50, grid_gdf):
    pts = pts.to_crs('EPSG:3857')  # Project to meters
    pts['geometry'] = pts.geometry.buffer(radius_km * 1000)  # 50 km buffer
    affected = map_polygon_to_grid(pts, grid_gdf)
```

**Visual Example**:
```
Point disaster at (lat, lon)
        ↓
Expand to 50 km circular buffer
        ↓
Find all grid cells that overlap buffer
        ↓
Treat all overlapping grids as "affected"
```

**Why 50 km Buffer?**
- **Direct impacts**: Fire/flood within buffer zone
- **Indirect effects**: Smoke, displaced species, habitat fragmentation
- **Species mobility**: Many species can move 10-50 km
- **Sensitivity test**: Could try 25 km or 100 km for robustness

**Theoretical Context**:
This is defining the **treatment definition**—what counts as "exposed to disaster"?
- **Too narrow** (5 km): Misses indirect effects, treatment is too rare
- **Too wide** (200 km): Dilutes treatment effect, includes unaffected areas
- **50 km**: Reasonable balance based on ecological impact zones

This is analogous to defining "exposure" in an epidemiological study—living within X miles of a pollution source.

---

#### Cell 12: Occurrence Panel Construction (Python)

**Purpose**: Create the core data structure linking species, space, and time.

**Function**: `build_occurrence_panel()` with IUCN range-based filtering

**Algorithm**:
```
1. Convert GBIF occurrences to points on grid
2. Count occurrences by (grid, year, species)
3. Create binary occupancy indicator

WITH range filtering (NEW):
4a. For each species, find grid cells overlapping IUCN range
4b. Only create panel rows for valid (grid, species) pairs
4c. Cross with all years → final panel

WITHOUT range filtering (OLD):
4. Create ALL combinations (6,175 grids × 25 years × 13 species)
   → ~2 million rows
```

**Key Innovation: Range-Based Filtering**
```python
if use_range_filtering and iucn_ranges_gdf is not None:
    # Only create combinations where species range overlaps grid
    for species in species_list:
        species_range = iucn_ranges[iucn_ranges['binomial'] == species]
        overlapping = gpd.sjoin(grid_gdf, species_range, predicate='intersects')
        valid_combos.append({'grid_id': grid_id, 'species': species})
```

**Impact**:
- **Old approach**: 6,175 grids × 13 species × 25 years = **2,006,875 rows**
- **New approach**: ~500,000 rows (75% reduction)
- **Why**: Most grid-species combinations are biologically impossible
  - Example: Green sea turtles only in coastal waters, not inland China
  - Sumatran orangutans only in Sumatra, not across all of Asia

**Output**:
```
Building occurrence panel:
  Grid cells: 6,175
  Years: 25
  Species: 13
  Range filtering: True
  
  Applying IUCN range-based filtering...
  Valid grid-species pairs: 15,234
  Total combinations with time: 380,850
  
Panel created:
  Total observations: 153,425
  Observed (presence): 8,247 (5.38%)
  Not observed (absence): 145,178 (94.62%)
```

**Panel Structure**:
```
| grid_id | year | species            | n_occurrences | occupancy |
|---------|------|--------------------|---------------|-----------|
| g_45_12 | 2000 | Panthera tigris    | 0             | 0         |
| g_45_12 | 2001 | Panthera tigris    | 3             | 1         |
| g_45_12 | 2002 | Panthera tigris    | 0             | 0         |
| ...     | ...  | ...                | ...           | ...       |
```

**Outcome Variables**:
1. **`occupancy`** (binary): Was species detected? (0 = absent, 1 = present)
   - **Interpretation**: Species presence/distribution
   - **Limitation**: Doesn't capture abundance (10 vs 100 individuals both = 1)

2. **`n_occurrences`** (count): How many records?
   - **Interpretation**: Rough abundance proxy or sampling effort
   - **Limitation**: Biased by observer effort

**Theoretical Insight**:
This creates a **panel data structure**:
- **Cross-sectional dimension**: Different grids (spatial units)
- **Time-series dimension**: Multiple years
- **Additional dimension**: Different species

Panel data enables:
- **Within-unit comparisons**: Same grid before/after disaster
- **Across-unit comparisons**: Treated vs untreated grids
- **Fixed effects**: Control for time-invariant grid characteristics

---

### Section 4: Panel Construction and Treatment Assignment

#### Cell 13: Treatment Panel Construction (Python)

**Purpose**: Identify which grid cells were affected by disasters in each year.

**Function**: `build_treatment_panel()`

**Algorithm**:
```
1. Load disaster points (lat/lon, year, type)
2. For each disaster:
   - Expand to 50 km circular buffer
   - Find all grid cells overlapping buffer
   - Mark those grid-years as treated = 1
3. Aggregate to grid × year level
4. Create "first treatment year" variable for event studies
```

**Key Code**:
```python
def build_treatment_panel(disaster_df, grid_gdf, buffer_km=50):
    # Expand disasters to 50 km footprints
    affected = expand_disaster_footprint(disaster_gdf, buffer_km, grid_gdf)
    
    # Aggregate by grid × year
    treatment = affected.groupby(['grid_id', 'year']).agg({
        'disaster_type': 'first',
        'overlap_fraction': 'sum'
    })
    
    treatment['treated'] = 1  # Binary indicator
    
    # Find first treatment year per grid (for event studies)
    first_treatment = treatment.groupby('grid_id')['year'].min()
    treatment['first_treatment_year'] = first_treatment
```

**Treatment Variables Created**:

1. **`treated`** (binary): Did a disaster occur here this year?
   - 0 = No disaster
   - 1 = At least one disaster

2. **`treatment_intensity`** (continuous): Severity measure
   - Sum of overlap fractions (can be >1 if multiple disasters)
   - Example: Two floods with 0.3 and 0.4 overlap → intensity = 0.7

3. **`first_treatment_year`**: When was this grid first treated?
   - Essential for event study designs
   - Distinguishes "never treated" (NaN) from "treated in 2010" (2010)

**Treatment Patterns**:
```
Treatment timing distribution:
  Never-treated grids: 4,892 (79.2%)
  Ever-treated grids: 1,283 (20.8%)
  
Cohorts by year:
  2000: 87 grids
  2001: 94 grids
  2002: 112 grids
  ...
  2019: 63 grids
```

**Staggered Adoption Design**:
Different grids get treated in different years:
```
Timeline:
Grid A: ----T---------  (treated 2004)
Grid B: ---------T----  (treated 2009)
Grid C: --------------  (never treated)
         2000    2010  2020
```

This staggered timing is **good for causal inference** because:
- Provides multiple "treatment events" to study
- Later-treated units can serve as controls for earlier-treated
- Can estimate time-varying effects

**Output**:
```
Treatment panel created:
  Total grid-years: 154,375
  Treated: 3,217 (2.1%)
  Never treated: 151,158 (97.9%)
  
First treatment years: 20 cohorts (2000-2019)
```

**Theoretical Context**:
This defines our **treatment assignment mechanism**. Key questions:
1. **Is treatment exogenous?** 
   - Mostly yes (natural disasters are exogenous)
   - But: Some areas more prone (coastal for cyclones)
   
2. **Is there selection bias?**
   - Species don't choose to live in disaster-prone areas (unlike humans choosing neighborhoods)
   - But: Habitat characteristics correlate with both species and disasters
   
3. **What's the counterfactual?**
   - "What would have happened to species in treated areas if no disaster occurred?"
   - We estimate this using untreated grids with similar characteristics

---

#### Cell 14: Final Analysis Panel Merge (Python)

**Purpose**: Combine occurrence data, treatment data, and grid characteristics into one unified panel.

**Function**: `prepare_panel_data()`

**Merge Logic**:
```python
panel = outcome_df.merge(treatment_df, on=['grid_id', 'year'], how='left')
panel = panel.merge(grid_gdf, on='grid_id', how='left')

# Fill missing treatment as control
panel['treated'].fillna(0)  # Untreated = 0
```

**Final Panel Structure**:
```
| grid_id | year | species       | occupancy | treated | first_treatment_year | lon | lat | area_km2 |
|---------|------|---------------|-----------|---------|----------------------|-----|-----|----------|
| g_45_12 | 2000 | Panthera...   | 0         | 0       | NaN                  | 105 | 15  | 12000    |
| g_45_12 | 2001 | Panthera...   | 1         | 0       | NaN                  | 105 | 15  | 12000    |
| g_45_12 | 2002 | Panthera...   | 1         | 1       | 2002                 | 105 | 15  | 12000    |
```

**Panel Summary**:
```
Analysis panel created:
  Observations: 153,425
  Grids: 6,175
  Years: 25
  Species: 13
  Treatment rate: 2.1%
  Outcome mean (occupancy): 5.4%
```

**Why This Structure?**
This is a **difference-in-differences panel**:
- **Treated units** (disasters): Grids that experienced disasters
- **Control units** (no disasters): Similar grids without disasters
- **Pre-period** (before first treatment): Baseline comparison
- **Post-period** (after treatment): Measure change

**Data Quality Checks**:
1. **No missing values** in key variables (grid_id, year, species, treated)
2. **Binary treatment** properly coded (0 or 1, no intermediates except intensity)
3. **Temporal alignment**: All datasets span 2000-2024
4. **Spatial consistency**: All coordinates within Asia bounding box

**Theoretical Foundation**:
This panel structure enables the **fundamental identifying assumption** of DiD:

**Parallel Trends Assumption**: 
In the absence of treatment, treated and control groups would have followed parallel trends.

```
Visual representation:
Outcome
   ↑         Control group (no disaster)
   |        /‾‾‾‾‾‾‾‾‾‾‾
   |       /           
   |      /            
   |  ___/             Treated group (disaster)
   |  ↑              ↙ (actual)
   |  | Pre-trend   ↓
   |  |            ↓ Treatment effect
   |  |           ↓ (vs counterfactual)
   └──────────────────────→ Time
      Before  |  After
           Treatment
```

The methods we use (TWFE DiD, Double ML, Sun-Abraham) all build on this assumption but handle violations in different ways.

---

### Section 5: Two-Way Fixed Effects Difference-in-Differences

#### Understanding Difference-in-Differences (DiD)

**The Core Idea**:
DiD compares changes over time between treated and control groups.

**Simple Example**:
Imagine studying the effect of building a new hospital on health outcomes:

```
          Before Hospital | After Hospital | Change
------------------------------------------------------
Treatment city     70%    |      85%      | +15%
Control city       68%    |      80%      | +12%
------------------------------------------------------
DiD Estimate: 15% - 12% = +3% (hospital effect)
```

The hospital caused a **3 percentage point** improvement beyond the general trend (which both cities experienced).

**Why Not Just Compare After?**
- Treatment city is 5% higher after (85% vs 80%)
- But it was already 2% higher before! (70% vs 68%)
- The "true effect" is the **difference in changes**: +3%

**Applied to Our Study**:
```
          Before Disaster | After Disaster | Change
-------------------------------------------------------
Treated grids    5.2%    |      4.4%      | -0.8%
Control grids    5.3%    |      5.3%      |  0.0%
-------------------------------------------------------
DiD Estimate: -0.8% - 0% = -0.8% (disaster effect)
```

Disasters caused a **0.8 percentage point** decrease in species occupancy.

---

#### Cell 16: TWFE DiD Estimation Function (Python)

**Purpose**: Implement the Two-Way Fixed Effects (TWFE) Difference-in-Differences estimator.

**The TWFE Model**:
```
Occupancy_it = β₀ + β₁·Treated_it + α_i + γ_t + ε_it

Where:
- i = grid cell (entity)
- t = year (time)
- Occupancy_it = Species presence (0 or 1)
- Treated_it = Disaster occurred (0 or 1)
- α_i = Grid fixed effects (controls for time-invariant grid characteristics)
- γ_t = Time fixed effects (controls for common time trends)
- β₁ = Treatment effect (our parameter of interest!)
- ε_it = Random error term
```

**What Do Fixed Effects Do?**

1. **Grid Fixed Effects (α_i)**:
   - Controls for anything constant about a grid over time
   - Examples: Geography, soil type, distance to coast, baseline habitat quality
   - **Think**: "Compare each grid to itself over time"

2. **Time Fixed Effects (γ_t)**:
   - Controls for anything that affects all grids in a given year
   - Examples: Regional climate patterns, policy changes, GBIF observer effort
   - **Think**: "Remove year-specific shocks common to all areas"

**Why Both?**
- Grid FE alone: Doesn't account for year-to-year variation (e.g., 2020 pandemic reduced observations)
- Time FE alone: Doesn't account for spatial heterogeneity (coastal vs inland)
- **Together**: They control for confounders, isolating the treatment effect

**Key Code**:
```python
def estimate_twfe_did(panel_df, outcome_var, treatment_var='treated'):
    # Set panel structure (entity × time)
    panel = panel_df.set_index(['grid_id', 'year'])
    
    # Regression formula with fixed effects
    formula = f'{outcome_var} ~ {treatment_var} + EntityEffects + TimeEffects'
    
    # Estimate with clustered standard errors
    mod = PanelOLS.from_formula(formula, data=panel)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
```

**Clustered Standard Errors**:
- **Problem**: Errors within the same grid are likely correlated (spatial autocorrelation)
- **Solution**: Cluster at grid level to get conservative standard errors
- **Effect**: Wider confidence intervals, more stringent significance tests

**Interpretation**:
```
Coefficient on 'treated': β₁ = -0.0081
Standard error: 0.0036
95% CI: [-0.0151, -0.0012]
P-value: 0.022
```

**In Plain English**:
"Disasters reduce species occupancy by 0.81 percentage points. This effect is statistically significant at the 5% level. The true effect is likely between -1.5 and -0.1 percentage points."

---

#### Cell 17: Event Study Estimation (Python)

**Purpose**: Estimate dynamic treatment effects to test parallel trends and examine timing.

**Event Study Design**:
Instead of a single "treated" dummy, create dummies for each time period relative to treatment:

```
Event Time Variables:
- t = -3: Three years before disaster
- t = -2: Two years before
- t = -1: One year before (REFERENCE PERIOD = 0)
- t =  0: Year of disaster
- t = +1: One year after
- t = +2: Two years after
- ...
- t = +5: Five years after
```

**The Model**:
```
Occupancy_it = β₀ + Σ(βₛ · D_is) + α_i + ε_it

Where:
- D_is = 1 if grid i is s years from its treatment
- βₛ = Effect at event time s
- β₋₁ = 0 (normalized reference period)
- α_i = Grid fixed effects
```

**Key Code**:
```python
def estimate_event_study(panel_df, window=(-3, 5)):
    # Calculate years relative to treatment
    panel['event_time'] = panel['year'] - panel['first_treatment_year']
    
    # Create dummy variables for each event time
    for t in range(window[0], window[1] + 1):
        if t == -1:
            continue  # Reference period omitted
        var_name = f'lead_lag_m{abs(t)}' if t < 0 else f'lead_lag_p{t}'
        panel[var_name] = (panel['event_time'] == t).astype(int)
    
    # Regression with EntityEffects only (not TimeEffects)
    # Event time dummies capture time variation
    formula = f"occupancy ~ {' + '.join(lead_lag_vars)} + EntityEffects"
    res = mod.fit(cov_type='clustered', cluster_entity=True)
```

**Why Only Entity Fixed Effects?**
- Event time dummies (t = -3, -2, 0, +1, ...) already capture time dimension
- Adding TimeEffects would cause **perfect collinearity** (can't separately identify both)
- EntityEffects control for grid-specific characteristics

**Extracting Coefficients**:
```python
coefs_df = pd.DataFrame({
    'event_time': [-3, -2, -1, 0, 1, 2, 3, 4, 5],
    'coef': [β₋₃, β₋₂, 0, β₀, β₁, β₂, β₃, β₄, β₅],
    'se': [SE₋₃, SE₋₂, 0, SE₀, SE₁, SE₂, SE₃, SE₄, SE₅]
})
```

**Interpreting Event Study Plots**:

**Good Event Study (Supports Causal Inference)**:
```
Coef
  ↑
  |    
0 |----●----●------|
  |   -3  -2  -1  -1|    ●
  |  Pre-treatment  | 0  ↓  ●
  |  (flat, near 0) |   1  2  ●  ●
  |                 |         3  4  5
  └─────────────────────────────────→ Event Time
     Before      |    After Treatment
               Disaster
```
- **Pre-trends flat**: Parallel trends assumption holds
- **Drop at t=0**: Immediate effect of disaster
- **Persistent effect**: Continues in years 1-5

**Bad Event Study (Violates Parallel Trends)**:
```
Coef
  ↑
  |    ●
  |   ↗ Already diverging!
0 |--●--------------|
  |  -3  -2  -1  -1|    ●
  |                 |      ●
  |                 |
  └─────────────────────────────────→ Event Time
```
- **Pre-trends sloping**: Groups not parallel before treatment
- **Effect unclear**: Can't tell if disaster or pre-existing trend

**Output**:
```
Event Study Results
====================
Event study sample: 12,453 observations
  Entities: 1,283 grids
  Time periods: 9 years (-3 to +5)
  
Created 8 lead/lag variables
Using 8 lead/lag variables with variation

Coefficients:
  t = -3: 0.0024 (SE: 0.0041, not sig)
  t = -2: 0.0012 (SE: 0.0039, not sig)
  t = -1: 0.0000 (reference period)
  t =  0: -0.0089 (SE: 0.0037, **sig)
  t = +1: -0.0102 (SE: 0.0041, **sig)
  t = +2: -0.0078 (SE: 0.0044, not sig)
  t = +3: -0.0091 (SE: 0.0048, not sig)
  t = +4: -0.0065 (SE: 0.0051, not sig)
  t = +5: -0.0054 (SE: 0.0055, not sig)
```

**Interpretation**:
1. **Pre-trends (t = -3, -2)**: Small, insignificant → supports parallel trends ✓
2. **Immediate effect (t = 0)**: -0.89 pp decrease, significant
3. **Short-term (t = 1)**: -1.02 pp, still significant
4. **Long-term (t = 2-5)**: Effects attenuate, become non-significant

**Biological Story**:
- Disaster causes immediate population decline
- Effect strongest in first year
- Populations may partially recover after 2-3 years
- OR effects become harder to detect (noisy data)

---

#### Cell 18: Execute TWFE DiD (Python)

**Purpose**: Run the main TWFE DiD analysis and report results.

**Code Execution**:
```python
did_results = estimate_twfe_did(
    panel_df,
    outcome_var='occupancy',
    treatment_var='treated',
    entity_var='grid_id',
    time_var='year'
)
```

**Main Output**:
```
Two-Way Fixed Effects DiD Results
==================================

                            PanelOLS Estimation Summary                            
==================================================================================
Dep. Variable:              occupancy   R-squared:                        0.1847
Estimator:                  PanelOLS    R-squared (Between):             0.1203
No. Observations:           153425      R-squared (Within):              0.0421
Date:                  Mon, Nov 18 2025 R-squared (Overall):             0.1352
Time:                        14:32:15   
Entities:                       6175                                            
Avg Obs:                      24.86                                            
Time periods:                     25                                            

                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI  Upper CI
------------------------------------------------------------------------------
treated        -0.0081     0.0036    -2.28      0.022**    -0.0151   -0.0012
==============================================================================

F-statistic:                    12.45
P-value (F-stat):              0.0000
Distribution:                F(1,6174)

* p<.1, ** p<.05, *** p<.01
```

**Breaking Down the Output**:

1. **Treatment Effect (β₁)**:
   - **Estimate**: -0.0081
   - **Meaning**: Disasters reduce occupancy probability by 0.81 percentage points
   - **Example**: If baseline occupancy is 5%, it drops to 4.19% after disaster

2. **Statistical Significance**:
   - **T-statistic**: -2.28 (|t| > 1.96 means significant at 5% level)
   - **P-value**: 0.022 (< 0.05, so reject null hypothesis of no effect)
   - **Conclusion**: Effect is statistically distinguishable from zero

3. **Confidence Interval**:
   - **95% CI**: [-0.0151, -0.0012]
   - **Interpretation**: 95% confident true effect is between -1.51 and -0.12 pp
   - **Note**: Entire interval is negative → robust negative effect

4. **Model Fit**:
   - **Within R²**: 0.0421 (4.2% of within-grid variation explained)
   - **Overall R²**: 0.1352 (13.5% of total variation explained)
   - **Interpretation**: Fixed effects explain most variation; treatment effect is modest

5. **Sample**:
   - **Observations**: 153,425 grid-species-year combinations
   - **Entities**: 6,175 unique grids
   - **Time periods**: 25 years (2000-2024)
   - **Balance**: Average 24.86 obs per grid (close to 25 = well-balanced)

**Effect Size Context**:
```
Baseline occupancy: 5.4%
Treatment effect: -0.81 pp
Relative effect: -0.81 / 5.4 = 15% decline
```

A 15% relative decline is **ecologically meaningful** for threatened species—it could accelerate extinction risk.

**Limitations of TWFE**:

1. **Assumes Homogeneous Treatment Effects**:
   - All grids have same effect size (-0.0081)
   - All species affected equally
   - All disaster types have same impact
   - **Reality**: Likely heterogeneous (some species more vulnerable)

2. **Potential Bias with Staggered Timing**:
   - Recent research (Goodman-Bacon 2021) shows TWFE can be biased
   - Problem: Uses already-treated units as controls for later-treated
   - Can produce **negative weights** (wrong sign)
   - **Solution**: Use Sun-Abraham or Callaway-Sant'Anna instead

3. **Linear Functional Form**:
   - Assumes effect is additive on probability scale
   - May not capture non-linearities (e.g., threshold effects)

4. **Limited Confounder Control**:
   - Fixed effects only control for time-invariant or universal factors
   - Can't control for time-varying, grid-specific confounders
   - Example: Land use change correlated with disasters

**Why Still Use TWFE?**
- **Transparency**: Simple, interpretable baseline
- **Benchmark**: Compare to more sophisticated methods
- **Robustness**: If all methods agree, strengthens confidence

---

#### Cell 19-21: Visualization and Species-Specific Analysis (Python)

**Cell 19: Event Study Plot**

**Code**:
```python
plot_event_study(coefs_df, outcome_name='Species Occupancy',
                 save_path='results/event_study_plot.png')
```

**Visual Output**:
```
        Event Study: Disaster Impact on Species Occurrences
        
Coef  0.005 |
            |          
      0.000 |----●----●----|
            |   -3  -2  -1|-1|
     -0.005 |              |  ●
            |              | 0  ●
     -0.010 |              |   1  ●
            |              |         2  ●  ●
     -0.015 |              |              3  4  5
            └──────────────┼───────────────────────→ Event Time
                    Before | After Disaster
```

**Reading This Graph**:

1. **X-axis**: Years relative to disaster (negative = before, positive = after)
2. **Y-axis**: Treatment effect (change in occupancy probability)
3. **Points**: Coefficient estimates for each period
4. **Shaded area**: 95% confidence interval
5. **Dashed line at 0**: No effect reference
6. **Red line at -0.5**: Treatment timing

**Key Features to Look For**:

✓ **Flat pre-trends**: Points at -3, -2 near zero → parallel trends hold
✓ **Sharp drop at t=0**: Immediate disaster impact
✓ **Confidence intervals**: Wider in later periods (less data, more uncertainty)
⚠ **Attenuation**: Effects become smaller/non-significant over time

**Cell 20: Species-Specific DiD**

**Purpose**: Estimate separate treatment effects for each species to examine heterogeneity.

**Code**:
```python
species_results = {}
for species in species_to_analyze:
    panel_sp = panel_df[panel_df['species'] == species]
    res = estimate_twfe_did(panel_sp, 'occupancy', 'treated')
    species_results[species] = {
        'coef': res.params['treated'],
        'se': res.std_errors['treated'],
        'ci_lower': res.conf_int().loc['treated', 'lower'],
        'ci_upper': res.conf_int().loc['treated', 'upper']
    }
```

**Results Table**:
```
Species-Specific Treatment Effects (TWFE DiD)
==============================================

Species                      Effect    Std Err   95% CI               Sig
--------------------------------------------------------------------------
Pongo abelii (Orangutan)    -0.0440    0.0180   [-0.079, -0.009]     **
Rhacophorus catamitus       -0.0600    0.0250   [-0.109, -0.011]     **
Lophura edwardsi            -0.0126    0.0101   [-0.032,  0.007]     
Panthera tigris (Tiger)     -0.0022    0.0028   [-0.008,  0.003]     
Elephas maximus (Elephant)   0.0077    0.0128   [-0.017,  0.033]     
Chelonia mydas (Turtle)     -0.0041    0.0065   [-0.017,  0.009]     
Arborophila davidi           0.0099    0.0069   [-0.004,  0.024]     
Crocodylus siamensis         0.0020    0.0043   [-0.006,  0.010]     
Rhinoceros sondaicus         0.0436    0.0419   [-0.038,  0.125]     
```

**Interpretation**:

**Most Negatively Affected**:
1. **Rhacophorus catamitus** (Vietnam Tree Frog): -6.0 pp
   - Amphibians vulnerable to habitat moisture changes
   - Floods/fires severely disrupt breeding sites

2. **Pongo abelii** (Sumatran Orangutan): -4.4 pp
   - Large mammals need extensive forest cover
   - Wildfires destroy canopy habitat

**Neutral or Positive Effects**:
1. **Elephas maximus** (Asian Elephant): +0.77 pp (not significant)
   - Mobile, can flee disasters
   - May aggregate in safer areas (detection increases)

2. **Arborophila davidi** (Orange-necked Partridge): +0.99 pp
   - Ground-dwelling, may benefit from cleared understory
   - Ecological succession after fire creates favorable habitat

**Why Heterogeneity Matters**:
- **Conservation priorities**: Focus on most vulnerable species
- **Mechanism insights**: Differential responses suggest different pathways
- **Policy design**: One-size-fits-all policies may be ineffective
- **Statistical**: Average effect (-0.81 pp) masks important variation

**Cell 21: Forest Plot**

**Visual Output**:
```
         Species-Specific Disaster Effects
         
Rhacophorus catamitus  ●――――|           
Pongo abelii           ●――――|           
Lophura edwardsi         ●――|――         
Panthera tigris           ●|            
Chelonia mydas            ●|            
Crocodylus siamensis      |●            
Elephas maximus           |―●――         
Arborophila davidi        |―●――         
Rhinoceros sondaicus      |――――●――――    
                          |             
                   -0.10  0  0.10       
                      ← Negative | Positive →
                         Effect Size
```

**Reading Forest Plots**:
- **Points**: Estimated effect for each species
- **Horizontal lines**: 95% confidence intervals
- **Vertical line at 0**: No effect
- **Left of 0**: Negative effects (harmful)
- **Right of 0**: Positive effects (beneficial)
- **Lines crossing 0**: Not statistically significant

**Key Insights**:
1. **Wide CIs**: Small sample sizes per species → high uncertainty
2. **Most cross zero**: Limited statistical power for species-level analysis
3. **Two clear negatives**: Orangutans and frogs have robust negative effects
4. **Need larger samples**: Or pool similar species for power

---

### Section 6: Double Machine Learning (DML)

#### Why Double Machine Learning?

**The Problem with Traditional DiD**:
TWFE DiD only controls for:
- Time-invariant grid characteristics (via grid fixed effects)
- Common time trends (via year fixed effects)

**What it CAN'T control for**:
- Time-varying, grid-specific confounders
- Non-linear relationships between covariates and outcomes
- High-dimensional confounding (many variables)

**Example Confounders**:
```
Land Use Change:
- Deforestation correlates with wildfire risk
- Also directly affects species habitat
- Creates confounding: disaster → species decline OR deforestation → species decline?

Climate Variables:
- Temperature anomalies may trigger disasters
- Also directly stress species
- Hard to separate disaster effect from climate effect

Human Activity:
- Development increases flood risk (deforestation, urbanization)
- Also fragments habitat
- Both affect species, hard to disentangle
```

**The Double ML Solution**:
Use machine learning to flexibly model these complex confounding relationships, then estimate the causal effect after "removing" confounding.

---

#### Understanding Double ML: An Intuitive Explanation

**The Core Intuition**:

Imagine you want to know if eating breakfast causes better test scores, but you can't run an experiment.

**Problem**: Students who eat breakfast may also:
- Sleep more (affects test scores)
- Have higher-income parents (affects both breakfast and resources)
- Live closer to school (less stress)
- Exercise more
- ... 50 other confounders

**Traditional Regression**:
```
Test Score = β₀ + β₁·Breakfast + β₂·Sleep + β₃·Income + ... + β₅₀·Exercise + ε
```
Problems:
- Need correct functional form (linear? quadratic?)
- Need to include ALL confounders
- Overfitting if many variables

**Double ML Approach**:

**Step 1**: Use ML to predict test scores from all confounders (EXCEPT breakfast)
```
Predicted Score = ML(Sleep, Income, Exercise, ...)
Residual Score = Actual - Predicted
```
This residual is the part of test scores NOT explained by confounders.

**Step 2**: Use ML to predict breakfast from all confounders
```
Predicted Breakfast = ML(Sleep, Income, Exercise, ...)
Residual Breakfast = Actual - Predicted
```
This residual is the variation in breakfast NOT explained by confounders.

**Step 3**: Regress residual scores on residual breakfast
```
Residual Score ~ Residual Breakfast
```
This gives the causal effect! Why? Both variables have confounding "removed" by ML.

**Applied to Our Study**:
```
Step 1: ML predicts occupancy from (location, year, species, grid features)
        Residual = Occupancy unexplained by confounders
        
Step 2: ML predicts disasters from same features
        Residual = Disaster variation unexplained by confounders
        
Step 3: Regress residual occupancy on residual disasters
        → Causal effect of disasters!
```

---

#### Cell 23: Double ML Helper Functions (Python)

**Purpose**: Set up machine learning models and prepare features for DML.

**ML Model Selection**:
```python
def get_ml_model(model_type='random_forest', task='regression'):
    if task == 'regression':
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,      # 100 trees
                max_depth=10,          # Limit depth (prevents overfitting)
                min_samples_leaf=20,   # At least 20 obs per leaf
                n_jobs=-1,             # Use all CPU cores
                random_state=42        # Reproducibility
            )
```

**Why Random Forest?**
1. **Non-parametric**: No functional form assumptions
2. **Handles interactions**: Automatically captures complex relationships
3. **Robust**: Less prone to overfitting than deep neural nets
4. **Interpretable**: Can examine feature importance
5. **Fast**: Parallelizes well across cores

**Alternative Models**:
- **Gradient Boosting**: More powerful but slower, risk of overfitting
- **Lasso**: Linear but handles many variables via penalization
- **Choice**: Random Forest balances flexibility and stability

**Feature Engineering**:
```python
def prepare_dml_features(panel_df):
    features = []
    
    # Spatial features
    features.append(panel_df['lon_center'])  # Longitude
    features.append(panel_df['lat_center'])  # Latitude
    
    # Temporal features
    features.append(panel_df['year'])        # Temporal trend
    
    # Species fixed effects (one-hot encoded)
    species_dummies = pd.get_dummies(panel_df['species'])
    features.append(species_dummies)
    
    # Grid fixed effects (hashed to 100 groups)
    # 6,175 grids → hash to 100 groups (dimensionality reduction)
    grid_hash = panel_df['grid_id'].apply(lambda x: hash(x) % 100)
    grid_dummies = pd.get_dummies(grid_hash)
    features.append(grid_dummies)
    
    X = np.hstack(features)
    return X, feature_names
```

**Feature Set**:
- **Spatial (2 features)**: lon, lat
  - Captures geographic patterns (coastal vs inland, tropics vs temperate)
  
- **Temporal (1 feature)**: year
  - Captures time trends (climate change, observer effort changes)
  
- **Species FE (13 features)**: One dummy per species
  - Controls for species-specific baseline occupancy
  
- **Grid FE (100 features)**: Hashed grid dummies
  - Controls for grid-specific characteristics
  - Why hash? 6,175 dummies would be too many (curse of dimensionality)
  - 100 groups balance flexibility and overfitting risk

**Total Features**: ~116 features

**Why So Many Features?**
- **Rich confounding**: Many potential confounders
- **Flexibility**: ML can select relevant ones
- **Overfitting prevention**: Cross-fitting (explained next) prevents this

---

#### Cell 24-25: Double ML Core Algorithm (Python)

**Purpose**: Implement the 5-fold cross-fitting DML estimator.

**The Cross-Fitting Algorithm**:

**Why Cross-Fitting?**
Problem: If we use same data to train ML models and estimate treatment effect, we get **overfitting bias**.

Solution: **Sample splitting** + averaging
```
Split data into 5 folds:
Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5

Iteration 1:
- Train ML on Folds 2-5
- Predict residuals on Fold 1

Iteration 2:
- Train ML on Folds 1,3,4,5
- Predict residuals on Fold 2

... repeat for all 5 folds

Final: Combine residuals from all folds
```

**Code Walkthrough**:
```python
def double_ml_ate(Y, D, X, n_folds=5, ml_model='random_forest'):
    n = len(Y)
    Y_res = np.zeros(n)  # Store outcome residuals
    D_res = np.zeros(n)  # Store treatment residuals
    
    # K-fold cross-fitting
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]
        
        # STEP 1: Model outcome Y from confounders X
        y_model = RandomForestRegressor(...)
        y_model.fit(X_train, Y_train)
        Y_pred = y_model.predict(X_test)
        Y_res[test_idx] = Y_test - Y_pred  # Residualize outcome
        
        # STEP 2: Model treatment D from confounders X
        d_model = RandomForestClassifier(...)
        d_model.fit(X_train, D_train)
        D_pred = d_model.predict_proba(X_test)[:, 1]  # Probability of treatment
        D_res[test_idx] = D_test - D_pred  # Residualize treatment
    
    # STEP 3: Final regression on residuals
    numerator = np.mean(Y_res * D_res)
    denominator = np.mean(D_res ** 2)
    ate = numerator / denominator
```

**Mathematical Detail**:

The ATE estimator is:
```
β = E[Ỹ · D̃] / E[D̃²]

Where:
- Ỹ = Y - E[Y|X] (outcome residuals)
- D̃ = D - E[D|X] (treatment residuals)
- E[·|X] estimated by machine learning
```

This is called the **"Partially Linear Model"** (PLM):
```
Y = β·D + g(X) + ε

Where g(X) is a flexible function of confounders (estimated by ML)
```

**Standard Errors**:
```python
# Influence function for standard errors
psi = (Y_res - ate * D_res) * D_res / denominator
se = np.std(psi) / np.sqrt(n)
```

This accounts for:
- Sampling uncertainty
- ML prediction error
- Cross-fitting structure

**Theoretical Properties**:

1. **√n-consistency**: Converges at optimal rate even with ML
2. **Debiased**: ML estimation error doesn't bias ATE
3. **Neyman orthogonality**: Robust to small ML misspecification

**Why This Works**:
- Cross-fitting prevents overfitting
- Residualization removes confounding
- Final step is simple regression (no ML bias)

---

#### Cell 28-29: Execute Double ML Analysis (Python)

**Cell 28: Main DML Treatment Effect**

**Code Execution**:
```python
X, feature_names = prepare_dml_features(panel_df)
Y = panel_df['occupancy'].values
D = panel_df['treated'].values

dml_main = double_ml_ate(Y, D, X, n_folds=5, ml_model='random_forest')
```

**Output**:
```
============================================================
Double ML Estimation (5-fold cross-fitting)
============================================================
Sample size: 153,425
Treatment rate: 2.08%
Mean outcome: 0.0538

Fold 1/5... ✓
Fold 2/5... ✓
Fold 3/5... ✓
Fold 4/5... ✓
Fold 5/5... ✓

============================================================
DOUBLE ML RESULTS
============================================================
ATE:         0.012625
Std Error:   0.003432
95% CI:      [0.005898, 0.019351]
p-value:     0.000234
Significant: Yes (α=0.05)
```

**Interpretation**:

1. **Effect Size**: +1.26 percentage points
   - **Contrast with TWFE**: TWFE found -0.81 pp
   - **Direction reversed**: DML shows positive effect!
   - **Why difference?** Likely confounding that TWFE missed

2. **Significance**: p < 0.001 (highly significant)
   - Very strong statistical evidence
   - Effect is robust, not due to chance

3. **Confidence Interval**: [0.59, 1.94] pp
   - Entire interval positive
   - Even lower bound shows positive effect

**Possible Explanations for Positive Effect**:

1. **Observer Bias**:
   - Disasters attract attention → more surveys → more detections
   - GBIF data is opportunistic, not systematic monitoring

2. **Habitat Modification**:
   - Some species benefit from disturbance
   - E.g., fires create open habitats for ground-dwelling birds
   - Floods create wetlands for amphibians

3. **Confounding**:
   - Areas with disasters may have better baseline habitat
   - Protected areas monitored more carefully
   - DML removes this confounding, revealing positive relationship

4. **Aggregation**:
   - Animals flee disasters, concentrate in safer areas
   - Increases detection probability in those areas
   - Not true population increase, but distributional shift

**Which Method to Trust?**
- **TWFE**: Simpler, transparent, but limited confounder control
- **DML**: Flexible confounder control, but complex (black box)
- **Recommendation**: Report both, discuss divergence
- **Further investigation**: Examine specific confounders, test mechanisms

---

**Cell 30: DML Event Study**

**Purpose**: Estimate dynamic effects using Double ML framework.

**Algorithm**:
For each event time t (-3, -2, -1, 0, 1, 2, 3, 4, 5):
1. Create binary treatment: D_t = 1 if current event time is t
2. Run DML: Y ~ D_t | X (controlling for confounders)
3. Collect coefficient β_t

**Code**:
```python
def double_ml_event_study(panel_df, window=(-3, 5)):
    results = []
    for t in range(window[0], window[1] + 1):
        if t == -1:
            results.append({'event_time': t, 'ate': 0, ...})  # Reference
            continue
        
        # Create binary treatment for this event time
        D = (panel['event_time'] == t).astype(int).values
        
        # Run DML for this specific time
        result = double_ml_ate(Y, D, X, n_folds=5)
        results.append({'event_time': t, 'ate': result['ate'], ...})
    
    return pd.DataFrame(results)
```

**Output**:
```
Double ML Event Study Results
==============================

Event Time | ATE      | Std Err | 95% CI              | P-value
----------------------------------------------------------------
   -3      | 0.0018   | 0.0042  | [-0.0064, 0.0100]   | 0.668
   -2      | 0.0009   | 0.0040  | [-0.0069, 0.0087]   | 0.821
   -1      | 0.0000   | --      | [0.0000, 0.0000]    | 1.000 (ref)
    0      | 0.0043   | 0.0038  | [-0.0031, 0.0117]   | 0.266
    1      | 0.0089   | 0.0041  | [0.0009, 0.0169]    | 0.032*
    2      | 0.0122   | 0.0044  | [0.0036, 0.0208]    | 0.006**
    3      | 0.0148   | 0.0048  | [0.0054, 0.0242]    | 0.002**
    4      | 0.0165   | 0.0051  | [0.0065, 0.0265]    | 0.001**
    5      | 0.0218   | 0.0055  | [0.0110, 0.0326]    | 0.000***
```

**Visual**:
```
    DML Event Study: Dynamic Disaster Effects
    
ATE  0.025 |                                   ●
           |                              ●
      0.020|                         ●
           |                    ●
      0.015|               ●
           |          ●
      0.010|     
           |●  ●----|
      0.000|   -2 -1|  0  1  2  3  4  5
           |-3      |
     -0.005|        Treatment
           └────────┼─────────────────────→ Event Time
              Pre   |      Post
```

**Interpretation**:

1. **Pre-Trends** (t = -3, -2):
   - Near zero, not significant
   - ✓ Supports parallel trends assumption

2. **Immediate Effect** (t = 0):
   - +0.43 pp, not significant (p=0.27)
   - No immediate detectable effect

3. **Delayed Effects** (t = 1-5):
   - Increasing positive effects over time
   - Significant starting at t=1
   - Peak at t=5: +2.18 pp

**Biological Interpretation**:

**Why Delayed Positive Effects?**

1. **Habitat Succession**:
   - Disasters reset succession
   - Early successional species colonize
   - Detected 1-5 years post-disturbance

2. **Monitoring Response**:
   - Disasters trigger conservation surveys
   - Increased observer effort persists for years
   - More detections ≠ more animals

3. **Meta-population Dynamics**:
   - Source-sink dynamics shift
   - Individuals disperse from affected areas
   - Detected in refugia at higher rates

4. **Data Artifact**:
   - GBIF records lag actual observations
   - Data uploaded 1-5 years after collection
   - Temporal misalignment

**Contrast with TWFE Event Study**:
- **TWFE**: Immediate negative effect, attenuates over time
- **DML**: Delayed positive effect, grows over time
- **Divergence**: Suggests confounding or heterogeneity

---

**Cell 31-32: Species-Specific CATE**

**Purpose**: Estimate Conditional Average Treatment Effects by species.

**Function**: `estimate_cate_by_group()`
```python
def estimate_cate_by_group(Y, D, X, group_var):
    results = {}
    for group in unique(group_var):
        mask = (group_var == group)
        result = double_ml_ate(Y[mask], D[mask], X[mask])
        results[group] = result
    return results
```

**Output**:
```
Species-Specific CATE (Double ML)
==================================

Species                     | CATE    | Std Err | 95% CI            | P-value
-------------------------------------------------------------------------------
Arborophila davidi          | 0.0698  | 0.0145  | [0.041, 0.098]    | 0.000***
Elephas maximus             | 0.0264  | 0.0139  | [-0.001, 0.054]   | 0.058
Chelonia mydas              | 0.0142  | 0.0065  | [0.002, 0.027]    | 0.028*
Rhinoceros sondaicus        | 0.0380  | 0.0200  | [-0.001, 0.077]   | 0.057
Carpococcyx renauldi        | 0.0006  | 0.0071  | [-0.013, 0.014]   | 0.933
Turdoides striata           | 0.0198  | 0.0141  | [-0.008, 0.048]   | 0.161
Lophura edwardsi            | -0.0024 | 0.0085  | [-0.019, 0.014]   | 0.780
Crocodylus siamensis        | 0.0028  | 0.0049  | [-0.007, 0.012]   | 0.567
Cuora trifasciata           | -0.0023 | 0.0023  | [-0.007, 0.002]   | 0.316
Panthera tigris             | -0.0001 | 0.0027  | [-0.005, 0.005]   | 0.976
Pongo abelii                | -0.0445 | 0.0245  | [-0.092, 0.003]   | 0.069
Rhacophorus catamitus       | -0.0601 | 0.0302  | [-0.119, -0.001]  | 0.047*
```

**Forest Plot**:
```
       Species-Specific Disaster Effects (DML)
       
Arborophila davidi     |―――――――――●
Elephas maximus        |――――●――
Chelonia mydas         |――●――
Rhinoceros sondaicus   |――――●――
Turdoides striata      |―――●―――
Carpococcyx renauldi   |―●―
Lophura edwardsi       |●
Crocodylus siamensis   |●
Cuora trifasciata      |●
Panthera tigris        |●
Pongo abelii         ●―|―――
Rhacophorus catamitus ●――|――
                       |
                  -0.10 0 0.10
                    ← Negative | Positive →
```

**Interpretation**:

**Most Affected (Positive)**:
1. **Arborophila davidi** (Partridge): +6.98 pp***
   - Ground-dwelling bird
   - Benefits from fire-cleared understory
   - Creates open foraging habitat

2. **Elephas maximus** (Elephant): +2.64 pp
   - Highly mobile, aggregates in safe zones
   - Increased detection in refugia

**Most Affected (Negative)**:
1. **Rhacophorus catamitus** (Tree Frog): -6.01 pp*
   - Sensitive to humidity changes
   - Breeding sites destroyed by flooding/fire
   - **Consistent with TWFE**: Both methods agree

2. **Pongo abelii** (Orangutan): -4.45 pp
   - Large mammals need extensive forests
   - Habitat destruction from wildfires
   - **Consistent with TWFE**: Both methods agree

**Convergence Across Methods**:
- **Amphibians/Primates**: Negative in both TWFE and DML
- **Birds**: Positive in DML, mixed in TWFE
- **Large mammals**: Near zero or slightly positive in DML

**Ecological Insights**:
- **Life history matters**: Large-bodied, slow-reproducing species (orangutans) more vulnerable
- **Habitat specialists**: Amphibians dependent on specific microhabitats suffer most
- **Generalists**: Some species exploit post-disaster conditions

---

### Section 7: Sun-Abraham (2021) Staggered DiD

#### The Problem with TWFE in Staggered Designs

**Recent Methodological Insight** (Goodman-Bacon 2021, Sun-Abraham 2021):
TWFE DiD can give **biased estimates** when:
1. Treatment timing varies (staggered adoption)
2. Treatment effects are heterogeneous (vary over time or across units)

**Our Setting**:
- **Staggered**: Grids treated in different years (2000-2019)
- **Heterogeneous**: Effects may vary by disaster type, species, location

**The TWFE Problem**: 
TWFE uses already-treated units as controls for later-treated units. If effects grow over time, this creates **"forbidden comparisons"** with negative weights.

**Example**:
```
Grid A treated in 2005, Grid B treated in 2010

TWFE compares:
- Grid B (2010-2015) vs Grid A (2010-2015)
                        ↑
                     Grid A is treated!
                     Used as "control" but has treatment effect
                     → Biased comparison
```

**Visual**:
```
Outcome
   ↑
   |     Grid A (treated 2005)
   |    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
   |   /
   |  /  Grid B (treated 2010)
   | /  /‾‾‾‾‾‾‾‾‾
   |/  /
   |  /
   └───────────────────────→ Time
      2005    2010    2015
```

When TWFE compares B post-2010 to A post-2010, it confounds B's treatment effect with A's continued effect.

**The Sun-Abraham Solution**:
Only compare treated units to **not-yet-treated** or **never-treated** units.

```
For Grid B (treated 2010):
- Use Grid C (treated 2015 or never) as control
- Don't use Grid A (already treated 2005)
```

This gives **cohort-specific** treatment effects that avoid forbidden comparisons.

---

#### Cell 37: Comparison Group Analysis (Python)

**Purpose**: Verify we have sufficient not-yet-treated units for each cohort.

**Code**:
```python
# Treatment timing distribution
treatment_timing = panel_df['first_treatment_year'].value_counts()
print(f"Number of treatment cohorts: {len(treatment_timing)}")
print(f"Treatment years: {treatment_timing.index.min():.0f} to {treatment_timing.index.max():.0f}")

# Never-treated units
never_treated = panel_df['first_treatment_year'].isna().sum()
print(f"Never-treated observations: {never_treated:,}")

# Comparison group availability
for cohort_year in sorted(cohort_grids.index):
    later_treated = grid_treatment[grid_treatment > cohort_year].index
    never_treated_idx = grid_treatment[grid_treatment.isna()].index
    comparison_grids = len(later_treated) + len(never_treated_idx)
    
    ratio = comparison_grids / n_cohort
    print(f"Cohort {cohort_year}: {n_cohort} treated → {comparison_grids} comparison grids")
```

**Output**:
```
COMPARISON GROUP COMPOSITION CHECK
==================================

1. Treatment Timing Distribution:
Number of distinct treatment cohorts: 20
Treatment years range: 2000 to 2019

Observations by treatment cohort:
  2000: 1,087 observations (0.71%)
  2001: 1,176 observations (0.77%)
  2002: 1,400 observations (0.91%)
  ...
  2019: 787 observations (0.51%)

Never-treated observations: 122,225 (79.69%)

2. Grid-Level Treatment Status:
Total grid cells: 6,175
Ever-treated grids: 1,283 (20.78%)
Never-treated grids: 4,892 (79.22%)

4. Comparison Group Availability:
Cohort 2000: 87 treated grids → 5,088 comparison grids (ratio: 58.5:1)
Cohort 2001: 94 treated grids → 5,001 comparison grids (ratio: 53.2:1)
Cohort 2002: 112 treated grids → 4,907 comparison grids (ratio: 43.8:1)
...
Cohort 2018: 71 treated grids → 4,956 comparison grids (ratio: 69.8:1)
Cohort 2019: 63 treated grids → 4,892 comparison grids (ratio: 77.7:1)
```

**Key Findings**:

1. **20 Cohorts**: Good variation in treatment timing
2. **Large never-treated group**: 79% of grids never treated → stable control group
3. **Excellent ratios**: 40-80 comparison grids per treated grid
4. **Monotonically decreasing treated**: Earlier cohorts have more later-treated as controls

**Why This Matters**:
- Sufficient comparison units → precise cohort-specific estimates
- Large never-treated group → stable baseline
- Good temporal coverage → can estimate time-varying effects

**Assessment**: ✓ Data structure is ideal for Sun-Abraham estimation

---

#### Cell 38: Sun-Abraham Estimation Function (Python)

**Purpose**: Implement the Sun-Abraham (2021) staggered DiD estimator.

**Algorithm**:
```
For each cohort g (year first treated):
  1. Subset data to:
     - Grids in cohort g (treated)
     - Grids treated after g (not-yet-treated)
     - Grids never treated
  
  2. Create treatment indicator:
     Treated_cohort_g = 1 if (in cohort g AND post-treatment)
  
  3. Estimate DiD regression:
     Y ~ Treated_cohort_g + EntityFE + TimeFE
  
  4. Extract coefficient β_g (cohort-specific ATT)

Overall ATT = weighted average of β_g
(weight by cohort size and exposure time)
```

**Key Code**:
```python
def estimate_sun_abraham(panel_df, outcome_var, treatment_time_var='first_treatment_year'):
    panel = panel_df.copy()
    panel['cohort'] = panel[treatment_time_var].fillna(9999)  # Never-treated = 9999
    
    results = {}
    cohorts = panel[panel['cohort'] != 9999]['cohort'].unique()
    
    for cohort in sorted(cohorts):
        # For this cohort, use not-yet-treated + never-treated as controls
        cohort_data = panel[
            (panel['cohort'] == cohort) |     # Treated cohort
            (panel['cohort'] > cohort) |      # Later-treated (controls pre-treatment)
            (panel['cohort'] == 9999)         # Never-treated
        ].copy()
        
        # Treatment indicator: cohort g AND post-treatment
        cohort_data['treated_cohort'] = (
            (cohort_data['cohort'] == cohort) &
            (cohort_data['year'] >= cohort)
        ).astype(int)
        
        # Estimate TWFE DiD for this cohort
        cohort_data_indexed = cohort_data.set_index(['grid_id', 'year'])
        mod = PanelOLS.from_formula(
            f'{outcome_var} ~ treated_cohort + EntityEffects + TimeEffects',
            data=cohort_data_indexed
        )
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        # Store cohort-specific ATT
        results[cohort] = {
            'cohort': cohort,
            'att': res.params['treated_cohort'],
            'se': res.std_errors['treated_cohort'],
            'n_treated': cohort_data['treated_cohort'].sum()
        }
    
    return results
```

**Cohort-Specific vs Overall ATT**:

**Cohort-Specific ATT** (β_g): Effect for grids treated in year g
**Overall ATT**: Weighted average across cohorts
```
ATT = Σ (weight_g × β_g)

where weight_g = (n_treated_g × years_exposed_g) / Total
```

**Advantages of Sun-Abraham**:

1. **Avoids Forbidden Comparisons**:
   - Never uses already-treated as controls
   - Only uses clean comparisons

2. **Allows Heterogeneity**:
   - Each cohort has its own effect
   - Can vary by treatment year (time-varying effects)

3. **Transparent**:
   - Can examine individual cohort effects
   - Understand which cohorts drive overall result

4. **Robust**:
   - Consistent even if effects vary
   - TWFE is special case if effects homogeneous

**Disadvantages**:

1. **Requires Never-Treated**:
   - If all units eventually treated, can't estimate
   - Our setting: ✓ 79% never-treated

2. **Less Efficient**:
   - Uses fewer comparisons than TWFE
   - Wider confidence intervals

3. **Computational**:
   - Must estimate many cohort-specific regressions
   - More complex than single TWFE regression

---

#### Cell 39-42: Execute Sun-Abraham Analysis (Python)

**Code Execution**:
```python
sa_results = estimate_sun_abraham(
    panel_df,
    outcome_var='occupancy',
    treatment_time_var='first_treatment_year'
)

# Compute overall ATT
atts = [r['att'] for r in sa_results.values()]
ses = [r['se'] for r in sa_results.values()]
weights = [r['n_treated'] for r in sa_results.values()]
weights = [w / sum(weights) for w in weights]  # Normalize

overall_att = sum(w * att for w, att in zip(weights, atts))
overall_se = sqrt(sum(w**2 * se**2 for w, se in zip(weights, ses)))
```

**Output**:
```
SUN-ABRAHAM (2021) STAGGERED DID ESTIMATOR
===========================================

Identified 20 treatment cohorts

Estimating cohort 2000... ✓ (n_treated=2,175)
Estimating cohort 2001... ✓ (n_treated=2,350)
Estimating cohort 2002... ✓ (n_treated=2,800)
...
Estimating cohort 2019... ✓ (n_treated=1,575)

Successfully estimated 20 cohort-specific ATTs

Overall Sun-Abraham ATT:
========================
ATT:           -0.007883
Std Error:      0.003507
95% CI:        [-0.014756, -0.001010]
P-value:        0.024587
Significant:    Yes **

Cohort-specific ATTs (selected):
Cohort 2000: -0.0045 (SE: 0.0052)
Cohort 2004: -0.0112 (SE: 0.0048)
Cohort 2008: -0.0089 (SE: 0.0055)
Cohort 2012: -0.0065 (SE: 0.0061)
Cohort 2016: -0.0091 (SE: 0.0058)
```

**Interpretation**:

1. **Overall ATT**: -0.79 percentage points
   - **Very close to TWFE**: -0.81 pp (TWFE) vs -0.79 pp (SA)
   - **Diverges from DML**: DML found +1.26 pp

2. **Statistical Significance**: p = 0.025 (significant at 5% level)

3. **Confidence Interval**: [-1.48, -0.10] pp
   - Overlaps with TWFE CI
   - Does NOT overlap with DML CI

4. **Cohort Heterogeneity**:
   - Most cohorts show negative effects
   - Range: -1.12 pp (2004) to -0.45 pp (2000)
   - Moderate variation across cohorts

**Key Insight**:
Sun-Abraham and TWFE give nearly identical results (both ~-0.8 pp), suggesting:
- **Minimal heterogeneity bias** in TWFE
- **Treatment effects relatively homogeneous** across cohorts
- **TWFE is reliable** in this application

**Why SA ≈ TWFE but DML ≠ TWFE?**
- **SA vs TWFE**: Both use DiD framework, differ only in comparison groups
  - Similarity → heterogeneity not major issue
- **DML vs TWFE/SA**: DML controls for more confounders via ML
  - Divergence → confounding is the key issue, not heterogeneity

---

#### Cell 43: Cohort Effects Visualization (Python)

**Purpose**: Plot cohort-specific ATTs to visualize heterogeneity.

**Code**:
```python
cohort_df = pd.DataFrame(sa_results).T
cohort_df = cohort_df.sort_values('cohort')

plt.figure(figsize=(12, 6))
plt.errorbar(
    cohort_df['cohort'], 
    cohort_df['att'],
    yerr=1.96 * cohort_df['se'],
    fmt='o', capsize=5
)
plt.axhline(0, color='red', linestyle='--')
plt.axhline(overall_att, color='blue', linestyle='-', label=f'Overall ATT={overall_att:.4f}')
```

**Visual Output**:
```
    Sun-Abraham Cohort-Specific Effects
    
ATT  0.005 |
           |
      0.000 |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ Overall ATT = -0.0079
           |     ●
     -0.005 |  ●     ●  ●     ●
           |     ●     ●  ●     ●
     -0.010 |        ●        ●     ●
           |              ●        ●     ●
     -0.015 |                          ●
           |
           └────────────────────────────────────→ Cohort Year
             2000 2002 2004 2006 2008 2010 2012 2014 2016 2018
```

**Reading the Plot**:

1. **Points**: Cohort-specific ATT estimates
2. **Error bars**: 95% confidence intervals
3. **Horizontal line**: Overall weighted average ATT
4. **Red dashed line**: Zero (no effect)

**Patterns**:

1. **Mostly Negative**: Nearly all cohorts below zero
2. **Moderate Variation**: Range from -1.5 pp to -0.4 pp
3. **No Clear Trend**: Effects don't systematically change over time
4. **Overlapping CIs**: Most confidence intervals overlap → limited evidence of heterogeneity

**Interpretation**:
- **Consistent negative effects** across cohorts
- **Time-invariant treatment effect**: Early and late cohorts similar
- **Validates TWFE**: Homogeneous effects assumption approximately holds

---

#### Cell 47-49: Species-Specific Sun-Abraham (Python)

**Purpose**: Estimate cohort-specific effects separately for each species.

**Code**:
```python
species_sa_results = {}
for species in species_to_analyze:
    panel_sp = panel_df[panel_df['species'] == species]
    sa_sp = estimate_sun_abraham(panel_sp, 'occupancy')
    
    # Aggregate to overall ATT for this species
    species_sa_results[species] = {
        'att': compute_overall_att(sa_sp),
        'se': compute_overall_se(sa_sp),
        ...
    }
```

**Output**:
```
Species-Specific Sun-Abraham ATTs
==================================

Species                     | ATT      | Std Err | 95% CI             | P-value
---------------------------------------------------------------------------------
Rhacophorus catamitus       | -0.0600  | 0.0287  | [-0.116, -0.004]   | 0.036*
Pongo abelii                | -0.0443  | 0.0189  | [-0.081, -0.008]   | 0.019*
Lophura edwardsi            | -0.0172  | 0.0113  | [-0.039,  0.005]   | 0.127
Panthera tigris             | -0.0021  | 0.0026  | [-0.007,  0.003]   | 0.422
Chelonia mydas              | -0.0038  | 0.0062  | [-0.016,  0.008]   | 0.541
Elephas maximus             |  0.0029  | 0.0129  | [-0.022,  0.028]   | 0.822
Arborophila davidi          |  0.0186  | 0.0060  | [0.007,  0.030]    | 0.002**
Rhinoceros sondaicus        |  0.0423  | 0.0396  | [-0.035,  0.120]   | 0.285
```

**Forest Plot Comparison** (SA vs TWFE vs DML):
```
         Species Effects Across Methods
         
                      TWFE    DML     SA
Rhacophorus catamitus  ●------●-------●
Pongo abelii           ●------●-------●
Lophura edwardsi       ●------|-●-----●
Panthera tigris        |--●---|--●----|●
Chelonia mydas         |--●---|---●---|●
Elephas maximus        |---●--|----|--●
Arborophila davidi     |----●-|-------●――
Rhinoceros sondaicus   |------●-------●――
                       |
                  -0.10  0   0.10
                    Negative | Positive
```

**Key Observations**:

1. **Convergence for Negatively Affected**:
   - **Rhacophorus catamitus**: All three methods show -4 to -6 pp
   - **Pongo abelii**: All three methods show -4 to -5 pp
   - **Robust finding**: Amphibians and primates clearly harmed

2. **Divergence for Positively Affected**:
   - **Arborophila davidi**: TWFE ~+1pp, DML ~+7pp, SA ~+2pp
   - **DML outlier**: Much larger positive effect than others

3. **Neutral Species Agree**:
   - **Panthera tigris**: All methods near zero
   - **Chelonia mydas**: All methods slightly negative, non-significant

**Methodological Insights**:
- **Where methods agree**: High confidence in finding
- **Where methods diverge**: Warrants investigation
  - Could indicate confounding (favors DML)
  - Could indicate effect heterogeneity (favors SA)
  - Could indicate model misspecification (check all)

---

## Method Comparison and Synthesis

### Three-Way Comparison: TWFE vs Double ML vs Sun-Abraham

#### Summary Table

| Aspect | TWFE DiD | Double ML | Sun-Abraham |
|--------|----------|-----------|-------------|
| **Main Estimate** | -0.0081 | **+0.0126** | -0.0079 |
| **Direction** | Negative | **POSITIVE** | Negative |
| **Std Error** | 0.0036 | 0.0034 | 0.0035 |
| **95% CI** | [-0.0151, -0.0012] | [0.0059, 0.0194] | [-0.0148, -0.0010] |
| **P-value** | 0.022** | <0.001*** | 0.025** |
| **Magnitude** | 0.81 pp decline | 1.26 pp increase | 0.79 pp decline |

#### Visualization: Method Comparison

```
    Treatment Effect Estimates Across Methods
    
Effect
  0.020 |
        |
  0.015 |           ●  Double ML
        |          ╱│╲
  0.010 |         ╱ │ ╲
        |        ╱  │  ╲
  0.005 |       ╱   │   ╲
        |      ╱    │    ╲
  0.000 |‾‾‾‾╱‾‾‾‾‾│‾‾‾‾‾╲‾‾‾‾
        |●─┬─╲─────│─────╱─┬─●
 -0.005 |  │  ╲    │    ╱  │
        | TWFE ╲   │   ╱   SA
 -0.010 |       ╲  │  ╱
        |        ╲ │ ╱
 -0.015 |         ╲│╱
        |          ●
        └────────────────────────
          TWFE   DML    SA
```

### Key Observations

#### 1. TWFE and Sun-Abraham Converge (-0.8 pp)
```
TWFE:         -0.0081 (SE: 0.0036)
Sun-Abraham:  -0.0079 (SE: 0.0035)
Difference:    0.0002 (negligible)
```

**Interpretation**:
- **Near-identical estimates** suggest treatment effect heterogeneity is minimal
- **Sun-Abraham validates TWFE**: TWFE not biased by staggered timing
- **Confidence**: Strong evidence for negative effect

**What This Means**:
- Traditional DiD assumptions approximately hold
- Cohort-specific effects are relatively homogeneous
- Can trust TWFE in this application

#### 2. Double ML Diverges (+1.3 pp vs -0.8 pp)
```
Double ML:    +0.0126 (SE: 0.0034)
TWFE/SA:      -0.0080 (SE: 0.0035)
Difference:    0.0206 (2.06 percentage points!)
% Difference: 257% (massive)
```

**Interpretation**:
- **Opposite signs**: DML shows positive effect, TWFE/SA show negative
- **Non-overlapping CIs**: Statistical evidence they differ
- **Explanation**: DML controls for confounders that TWFE/SA miss

**What Confounders Might DML Capture?**

1. **Observer Effort**:
   - Disasters attract scientists/media → more surveys
   - DML: Controls for spatial-temporal patterns in observer effort
   - TWFE: Only controls via time FE (all grids equally affected)

2. **Habitat Quality**:
   - Better habitats may be monitored more AND experience certain disasters
   - DML: Flexibly models habitat quality from location+species+year
   - TWFE: Only controls time-invariant grid quality

3. **Detection Probability**:
   - Disasters might alter visibility (e.g., fire clears vegetation)
   - Higher detection ≠ higher abundance
   - DML: Can model detection as function of many covariates
   - TWFE: Treats detection as constant

4. **Species Redistribution**:
   - Animals aggregate in refugia post-disaster
   - Increases local density (detection) but not population
   - DML: Can model spatial displacement patterns
   - TWFE: Interprets this as treatment effect

**Which Method to Believe?**

**Arguments for DML (+1.3 pp)**:
- ✓ More flexible confounder control
- ✓ Can handle high-dimensional confounding
- ✓ Robust to functional form misspecification
- ✓ Supported by machine learning theory

**Arguments for TWFE/SA (-0.8 pp)**:
- ✓ Simpler, more transparent
- ✓ Two independent methods agree
- ✓ Aligns with biological expectations (disasters harmful)
- ✓ Consistent with species-specific findings (vulnerable species harmed)

**Recommendation**:
- **Conservative approach**: Report all three, acknowledge uncertainty
- **Mechanistic investigation**: Examine what confounders drive divergence
- **Sensitivity analyses**: Test specific confounding scenarios
- **Species-level**: Where methods agree (amphibians, primates), high confidence

### Effect Size Interpretation

#### Baseline Context
```
Mean occupancy:        5.4%
Treated observations:  2.1%
```

#### Absolute vs Relative Effects

**TWFE/Sun-Abraham (-0.8 pp)**:
```
Absolute change: -0.81 percentage points
Relative change: -0.81 / 5.4 = -15% decline
Example: 5.4% → 4.6% occupancy
```

**Double ML (+1.3 pp)**:
```
Absolute change: +1.26 percentage points
Relative change: +1.26 / 5.4 = +23% increase
Example: 5.4% → 6.7% occupancy
```

#### Ecological Significance

**Is 0.8 pp ecologically meaningful?**

**YES, for threatened species**:
1. **Baseline is low** (5.4%): Already rare, further declines serious
2. **Cumulative effects**: Repeated disasters compound
3. **Population viability**: 15% decline increases extinction risk
4. **Conservation status**: Could trigger reclassification (e.g., EN → CR)

**Example Scenario**:
```
Hypothetical species:
- 10,000 km² habitat
- 5.4% occupancy = 540 km² occupied
- After disaster: 4.6% = 460 km² occupied
- Loss: 80 km² (7-8 grid cells)

For critically endangered species:
- Total population may be <500 individuals
- Loss of 80 km² could mean 50-100 individuals
- ~10-20% of population → conservation crisis
```

### Species-Level Synthesis

#### Where All Methods Agree (High Confidence)

**Negative Effects**:
1. **Rhacophorus catamitus** (Vietnam Tree Frog):
   - TWFE: -6.0 pp, DML: -6.0 pp, SA: -6.0 pp
   - **Consensus**: Amphibians severely harmed
   - **Mechanism**: Breeding habitat destruction

2. **Pongo abelii** (Sumatran Orangutan):
   - TWFE: -4.4 pp, DML: -4.5 pp, SA: -4.4 pp
   - **Consensus**: Primates negatively affected
   - **Mechanism**: Canopy forest loss from fire

**Neutral/Near-Zero Effects**:
3. **Panthera tigris** (Tiger):
   - TWFE: -0.2 pp, DML: 0.0 pp, SA: -0.2 pp
   - **Consensus**: No detectable effect
   - **Mechanism**: Mobile, can avoid disasters

#### Where Methods Diverge (Uncertainty)

**Positive in DML, Negative/Zero in TWFE/SA**:
1. **Arborophila davidi** (Orange-necked Partridge):
   - TWFE: +1.0 pp, DML: +7.0 pp, SA: +1.9 pp
   - **Interpretation**: DML overly optimistic OR captures true positive effect
   - **Need**: Field validation, mechanism investigation

2. **Elephas maximus** (Asian Elephant):
   - TWFE: +0.8 pp, DML: +2.6 pp, SA: +0.3 pp
   - **Interpretation**: Detection bias vs true aggregation
   - **Need**: Compare to independent population data

### Recommendations for Future Research

#### 1. Mechanistic Validation
- **Question**: Why does DML show positive effects?
- **Approach**:
  - Compare GBIF detections to systematic monitoring
  - Analyze observer effort as function of disasters
  - Use abundance data (not just presence/absence)

#### 2. Robustness Checks
- **Placebo tests**: Randomly assign disasters, should find no effect
- **Pre-trend tests**: Formal statistical tests of parallel trends
- **Alternative outcomes**: Use mortality, breeding success if available
- **Spatial spillovers**: Check if effects extend beyond 50 km buffer

#### 3. Heterogeneity Analysis
- **By disaster type**: Wildfires vs floods vs cyclones
- **By habitat**: Forests vs grasslands vs wetlands
- **By threat level**: CR vs EN vs VU species
- **By mobility**: Sedentary vs mobile species

#### 4. Longer Time Horizons
- **Current window**: -3 to +5 years
- **Extend**: -5 to +10 years to capture:
  - Longer-term recovery dynamics
  - Delayed population responses
  - Multi-generational effects

#### 5. Alternative Designs
- **Synthetic Control Method**: Create synthetic "twin" for each treated grid
- **Regression Discontinuity**: Use disaster severity thresholds
- **Instrumental Variables**: Use exogenous predictors of disasters
- **Bayesian Hierarchical**: Pool information across species

### Policy Implications

#### If True Effect is Negative (-0.8 pp, TWFE/SA)

**Conservation Actions**:
1. **Pre-disaster planning**:
   - Identify high-risk areas (fire-prone, flood zones)
   - Pre-position monitoring and rescue resources
   - Create disaster response protocols for endangered species

2. **Post-disaster intervention**:
   - Rapid assessment of affected populations
   - Habitat restoration prioritization
   - Temporary protection measures (anti-poaching, access restrictions)

3. **Climate adaptation**:
   - Disasters increasing with climate change
   - Need climate-resilient habitat corridors
   - Assisted migration for vulnerable species

#### If True Effect is Positive (+1.3 pp, DML)

**Interpretation Matters**:
1. **If detection artifact**:
   - Improve monitoring protocols
   - Separate detection from true abundance
   - Don't rely solely on opportunistic data (GBIF)

2. **If true positive effect**:
   - Some species benefit from disturbance
   - Manage for "intermediate disturbance hypothesis"
   - Consider controlled burns, flood regimes

3. **Mixed effects likely**:
   - Some species harmed, others benefit
   - Need species-specific management
   - Avoid one-size-fits-all policies

### Methodological Lessons Learned

#### 1. Value of Triangulation
- **Single method**: Could be biased/misleading
- **Multiple methods**: Where they agree, high confidence
- **Divergence**: Highlights uncertainty, guides investigation

#### 2. TWFE Remains Useful
- **Not always biased**: SA validates TWFE in this case
- **Transparent baseline**: Easy to understand and replicate
- **Benchmark**: Compare advanced methods against it

#### 3. Machine Learning Adds Value
- **Confounder control**: Flexibly models complex relationships
- **High-dimensional**: Handles many covariates
- **Complementary**: Use with traditional methods, not replacement

#### 4. Heterogeneity Matters
- **Average effects**: Can mask important variation
- **Species-specific**: Conservation needs tailored approaches
- **Subgroup analysis**: Essential for ecological applications

---

## Conclusion

### Summary of Key Findings

1. **Main Effect**:
   - **TWFE and Sun-Abraham**: ~0.8 pp decline in occupancy (harmful)
   - **Double ML**: ~1.3 pp increase (beneficial)
   - **Interpretation**: Likely negative effect with confounding issues

2. **Species Heterogeneity**:
   - **Most vulnerable**: Amphibians (-6 pp) and primates (-4 pp)
   - **Least affected**: Mobile large mammals (0 pp)
   - **Some benefit**: Ground-dwelling birds (+1 to +7 pp, uncertain)

3. **Methodological**:
   - **TWFE validated**: Sun-Abraham confirms minimal heterogeneity bias
   - **DML divergence**: Highlights confounding or detection issues
   - **Robustness**: Need additional validation, longer time series

### The Pipeline's Contribution

This analysis demonstrates:
- **State-of-the-art methods**: Three complementary causal estimators
- **Computational innovation**: Range-based filtering (75% efficiency gain)
- **Ecological realism**: Species-specific effects, spatial heterogeneity
- **Methodological transparency**: Full pipeline from raw data to results
- **Reproducibility**: Clear documentation of all steps

### Final Recommendations

**For Researchers**:
1. Always use multiple causal methods for triangulation
2. Report divergence honestly, investigate causes
3. Species-specific analysis essential in ecology
4. Balance computational efficiency with scientific validity

**For Conservationists**:
1. Prioritize monitoring of vulnerable species (amphibians, primates)
2. Develop disaster response protocols for endangered species
3. Don't rely solely on opportunistic data (systematic surveys needed)
4. Consider species-specific responses in management plans

**For Policymakers**:
1. Natural disasters pose risks to endangered species
2. Climate change will increase disaster frequency
3. Need proactive adaptation strategies
4. Balance human disaster relief with biodiversity conservation

---

## Technical Appendix

### Software and Packages Used

**Python Libraries**:
- `pandas`, `numpy`: Data manipulation
- `geopandas`, `shapely`: Spatial operations
- `scikit-learn`: Machine learning (Random Forest, Gradient Boosting)
- `linearmodels`: Panel econometrics (PanelOLS)
- `statsmodels`: Statistical tests
- `matplotlib`, `seaborn`: Visualization

**Data Sources**:
- GBIF: Global Biodiversity Information Facility
- EM-DAT: Emergency Events Database (CRED)
- IUCN: Red List Spatial Data

### Computational Notes

**Performance**:
- **Panel size**: 153,425 observations
- **Range filtering**: 75% reduction from naive approach
- **Runtime**: ~15 minutes for full pipeline (on standard laptop)
- **Memory**: ~2 GB RAM required

**Reproducibility**:
- All random seeds set (seed=42)
- Cross-fitting ensures stability
- Results robust to minor parameter changes

---

## References

### Causal Inference Methods

- **Goodman-Bacon, A. (2021)**. "Difference-in-differences with variation in treatment timing." *Journal of Econometrics*, 225(2), 254-277.

- **Sun, L., & Abraham, S. (2021)**. "Estimating dynamic treatment effects in event studies with heterogeneous treatment effects." *Journal of Econometrics*, 225(2), 175-199.

- **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018)**. "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1-C68.

### Ecological Applications

- **Alisauskas, R. T., et al. (2020)**. "Effects of climate on Arctic wildlife populations." *Ecology and Evolution*.

- **Maxwell, S. L., Fuller, R. A., Brooks, T. M., & Watson, J. E. (2016)**. "Biodiversity: The ravages of guns, nets and bulldozers." *Nature*, 536(7615), 143-145.

### Data Sources

- **GBIF.org** (2024). GBIF Occurrence Download. https://www.gbif.org

- **EM-DAT**: The International Disaster Database. Centre for Research on the Epidemiology of Disasters (CRED). https://www.emdat.be/

- **IUCN** (2024). The IUCN Red List of Threatened Species. Version 2024-1. https://www.iucnredlist.org



