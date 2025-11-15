# -*- coding: utf-8 -*-
"""
Climate Covariates Fetcher - this one is seralised incase threadpoolextractor doesnt work (caused some random rate limit issues)

safest code if all else fails !!!!!!!!!!!!


Fetches NDVI, Temperature, and Precipitation data for analysis

HOW TO RUN:
===========
1. Install dependencies:
   pip install pandas numpy requests earthengine-api tqdm

2. Authenticate Google Earth Engine (one-time setup):
   earthengine authenticate
   Follow the URL, login, and authorize

3. Configure settings in Config class:
   - BASE_DIR: Your data directory path
   - PROJECT_ID: Your Google Earth Engine project ID
   - GRID_SIZE_KM: Desired grid resolution in kilometers (e.g., 11, 25, 50)
   - SRC_CSV: Path to your input CSV file

4. Ensure your source CSV has these columns:
   - lat_center, lon_center: Coordinates of each location
   - year: Year of observation

5. Run the script:
   python climate_fetcher.py

The script will create checkpoint and cache files to resume interrupted runs.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import requests
import ee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Configuration
# ============================================================

class Config:
    BASE_DIR = "./data"  # Change to your directory
    PROJECT_ID = "practice-ada"  # Your Google Earth Engine project ID
    
    # ===== GRID SIZE CONFIGURATION =====
    # Specify desired grid size in kilometers
    # This will be used to round coordinates and aggregate data
    GRID_SIZE_KM = 25  # Change this to 11, 25, 50, etc.
    
    # Minimum resolutions for each data source (in km)
    MIN_NDVI_RES_KM = 0.25  # MODIS NDVI minimum: 250m = 0.25km
    MIN_TEMP_PRECIP_RES_KM = 25  # Open-Meteo minimum: ~25km
    
    # Calculate effective resolutions (use minimum if grid is smaller)
    NDVI_RES_KM = max(GRID_SIZE_KM, MIN_NDVI_RES_KM)
    TEMP_PRECIP_RES_KM = max(GRID_SIZE_KM, MIN_TEMP_PRECIP_RES_KM)
    
    # Convert km to degrees (approximate: 1 degree ≈ 111 km at equator)
    NDVI_RES_DEG = NDVI_RES_KM / 111.0
    TEMP_PRECIP_RES_DEG = TEMP_PRECIP_RES_KM / 111.0
    
    SRC_CSV = f"{BASE_DIR}/analysis_panel.csv"
    CHECKPOINT_CSV = f"{BASE_DIR}/climate_checkpoint.csv"
    
    GEE_CACHE = f"{BASE_DIR}/gee_cache.json"
    OM_CACHE = f"{BASE_DIR}/om_cache.json"
    
    # Column names (dynamically include resolution)
    NDVI_COL = f"ndvi_mean_(0–1)_({NDVI_RES_KM}x{NDVI_RES_KM}km)"
    TEMP_COL = f"temp_mean_(°C)_({TEMP_PRECIP_RES_KM}x{TEMP_PRECIP_RES_KM}km)"
    PRECIP_COL = f"precip_sum_(mm/year)_({TEMP_PRECIP_RES_KM}x{TEMP_PRECIP_RES_KM}km)"
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print(f"\n⚙️  CONFIGURATION")
        print(f"{'='*50}")
        print(f"Desired Grid Size: {cls.GRID_SIZE_KM} km")
        print(f"\nEffective Resolutions:")
        print(f"  NDVI:         {cls.NDVI_RES_KM} km ({cls.NDVI_RES_DEG:.4f}°)")
        print(f"  Temp/Precip:  {cls.TEMP_PRECIP_RES_KM} km ({cls.TEMP_PRECIP_RES_DEG:.4f}°)")
        if cls.GRID_SIZE_KM < cls.MIN_NDVI_RES_KM:
            print(f"\n⚠️  Grid size {cls.GRID_SIZE_KM}km < NDVI min ({cls.MIN_NDVI_RES_KM}km)")
            print(f"   Using NDVI minimum: {cls.NDVI_RES_KM}km")
        if cls.GRID_SIZE_KM < cls.MIN_TEMP_PRECIP_RES_KM:
            print(f"\n⚠️  Grid size {cls.GRID_SIZE_KM}km < Temp/Precip min ({cls.MIN_TEMP_PRECIP_RES_KM}km)")
            print(f"   Using Temp/Precip minimum: {cls.TEMP_PRECIP_RES_KM}km")
        print(f"{'='*50}\n")

# ============================================================
# Utility Functions
# ============================================================

def initialize_environment():
    """Initialize Earth Engine and create necessary directories"""
    os.makedirs(Config.BASE_DIR, exist_ok=True)
    
    # Initialize Earth Engine (run `earthengine authenticate` first in terminal)
    try:
        ee.Initialize(project=Config.PROJECT_ID)
        print("✅ Earth Engine initialized")
    except Exception as e:
        print(f"❌ Earth Engine initialization failed: {e}")
        print("Run 'earthengine authenticate' in terminal first")
        raise

def load_or_create_checkpoint():
    """Load checkpoint CSV or create from source"""
    if not os.path.exists(Config.CHECKPOINT_CSV):
        df = pd.read_csv(Config.SRC_CSV)
        
        # Add covariate columns
        for col in [Config.NDVI_COL, Config.TEMP_COL, Config.PRECIP_COL]:
            if col not in df.columns:
                df[col] = None
        
        df.to_csv(Config.CHECKPOINT_CSV, index=False)
        print(f"✅ Created checkpoint: {Config.CHECKPOINT_CSV}")
    else:
        df = pd.read_csv(Config.CHECKPOINT_CSV)
        print(f"✅ Loaded checkpoint: {Config.CHECKPOINT_CSV}")
    
    return df

def load_or_create_cache(cache_path):
    """Load cache file or create empty one"""
    if not os.path.exists(cache_path):
        with open(cache_path, "w") as f:
            json.dump({}, f)
        return {}
    
    with open(cache_path, "r") as f:
        return json.load(f)

def save_cache(cache, cache_path):
    """Save cache to file"""
    with open(cache_path, "w") as f:
        json.dump(cache, f)

# ============================================================
# GEE NDVI Fetcher
# ============================================================

def round_coord_gee(lat, lon):
    """Round coordinates to NDVI resolution grid"""
    # CHANGED: Now uses configurable resolution instead of hardcoded 0.1 degree
    deg = Config.NDVI_RES_DEG
    return (round(lat / deg) * deg, round(lon / deg) * deg)

def fetch_ndvi(lat, lon, year, max_retries=3):
    """Fetch NDVI for a location and year"""
    delay = 10
    # CHANGED: Buffer size now calculated from NDVI resolution
    # Buffer should cover half the grid cell to capture the area
    buffer_size = Config.NDVI_RES_KM * 1000 / 2  # Convert km to meters, half for radius
    
    for attempt in range(max_retries):
        try:
            point = ee.Geometry.Point(lon, lat)
            
            ndvi_coll = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select("NDVI")
            )
            
            if ndvi_coll.size().getInfo() > 0:
                # CHANGED: Using dynamic buffer size based on grid resolution
                ndvi_mean = ndvi_coll.mean().reduceRegion(
                    ee.Reducer.mean(), point.buffer(buffer_size), 250
                ).get("NDVI").getInfo()
                
                if ndvi_mean is not None:
                    return ndvi_mean / 10000
            
            return None
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"⏸️ Rate limit - sleeping {delay}s...")
                time.sleep(delay)
                delay *= 2
            elif attempt == max_retries - 1:
                print(f"⚠️ NDVI fetch failed: {e}")
                return None

def process_ndvi(df, cache):
    """Process NDVI for all unique year-location pairs"""
    print("\n🌿 Starting NDVI fetch...")
    
    # Add rounded coordinates
    df["gee_lat"], df["gee_lon"] = zip(
        *df.apply(lambda r: round_coord_gee(r["lat_center"], r["lon_center"]), axis=1)
    )
    
    # Get unique points per year
    unique_points = (
        df.groupby("year")[["gee_lat", "gee_lon"]]
        .apply(lambda x: list(set([tuple(i) for i in x.values.tolist()])))
        .to_dict()
    )
    
    for year in sorted(unique_points.keys(), reverse=True):
        coords = unique_points[year]
        print(f"\n📅 Year {year} - {len(coords)} unique points")
        
        for i in tqdm(range(0, len(coords), 50), desc=f"Year {year}"):
            batch = coords[i:i+50]
            
            for lat, lon in batch:
                cache_key = f"{year},{lat},{lon}"
                mask = (df["year"] == year) & (df["gee_lat"] == lat) & (df["gee_lon"] == lon)
                
                if cache_key in cache:
                    # Use cached value
                    df.loc[mask, Config.NDVI_COL] = cache[cache_key]
                else:
                    # Fetch new value
                    ndvi = fetch_ndvi(lat, lon, year)
                    cache[cache_key] = ndvi
                    df.loc[mask, Config.NDVI_COL] = ndvi
            
            # Save progress
            df.to_csv(Config.CHECKPOINT_CSV, index=False)
            save_cache(cache, Config.GEE_CACHE)
            time.sleep(2)
    
    return df, cache

# ============================================================
# Open-Meteo Temperature & Precipitation Fetcher
# ============================================================

def round_coord_om(lat, lon):
    """Round coordinates to Temperature/Precipitation resolution grid"""
    # CHANGED: Now uses configurable resolution instead of hardcoded 0.25 degree
    deg = Config.TEMP_PRECIP_RES_DEG
    return (round(lat / deg) * deg, round(lon / deg) * deg)

def fetch_temp_precip(lat, lon, year):
    """Fetch temperature and precipitation for a location and year"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "UTC"
    }
    
    try:
        r = requests.get(url, params=params)
        if r.status_code == 429:
            raise RuntimeError("Open-Meteo rate limit hit")
        r.raise_for_status()
        
        js = r.json().get("daily", {})
        
        # Calculate mean temperature
        temps = [
            (a + b) / 2 
            for a, b in zip(
                js.get("temperature_2m_max", []), 
                js.get("temperature_2m_min", [])
            ) 
            if a is not None and b is not None
        ]
        temp_mean = np.mean(temps) if temps else None
        
        # Sum precipitation
        precip_sum = np.sum(js.get("precipitation_sum", [])) if "precipitation_sum" in js else None
        
        return temp_mean, precip_sum
        
    except Exception as e:
        print(f"⚠️ Open-Meteo error: {e}")
        return None, None

def process_temp_precip(df, cache):
    """Process temperature and precipitation for all unique year-location pairs"""
    print("\n🌡️ Starting Temperature & Precipitation fetch...")
    
    # Add rounded coordinates
    df["om_lat"], df["om_lon"] = zip(
        *df.apply(lambda r: round_coord_om(r["lat_center"], r["lon_center"]), axis=1)
    )
    
    # Get unique points per year
    unique_points = (
        df.groupby("year")[["om_lat", "om_lon"]]
        .apply(lambda x: list(set([tuple(i) for i in x.values.tolist()])))
        .to_dict()
    )
    
    for year in sorted(unique_points.keys(), reverse=True):
        coords = unique_points[year]
        print(f"\n📅 Year {year} - {len(coords)} unique points")
        
        for i in tqdm(range(0, len(coords), 50), desc=f"Year {year}"):
            batch = coords[i:i+50]
            
            for lat, lon in batch:
                cache_key = f"{year},{lat},{lon}"
                mask = (df["year"] == year) & (df["om_lat"] == lat) & (df["om_lon"] == lon)
                
                if cache_key in cache:
                    # Use cached values
                    temp, precip = cache[cache_key]
                    df.loc[mask, Config.TEMP_COL] = temp
                    df.loc[mask, Config.PRECIP_COL] = precip
                else:
                    # Fetch new values
                    temp, precip = fetch_temp_precip(lat, lon, year)
                    cache[cache_key] = (temp, precip)
                    df.loc[mask, Config.TEMP_COL] = temp
                    df.loc[mask, Config.PRECIP_COL] = precip
            
            # Save progress
            df.to_csv(Config.CHECKPOINT_CSV, index=False)
            save_cache(cache, Config.OM_CACHE)
            time.sleep(1)
    
    return df, cache

# ============================================================
# Main Execution
# ============================================================

def main():
    """Main execution function"""
    print("🚀 Starting Climate Covariates Fetcher\n")
    
    # Display configuration
    Config.print_config()
    
    # Initialize
    initialize_environment()
    df = load_or_create_checkpoint()
    
    # Load caches
    gee_cache = load_or_create_cache(Config.GEE_CACHE)
    om_cache = load_or_create_cache(Config.OM_CACHE)
    
    # Fetch NDVI
    df, gee_cache = process_ndvi(df, gee_cache)
    
    # Fetch Temperature & Precipitation
    df, om_cache = process_temp_precip(df, om_cache)
    
    # Final save
    df.to_csv(Config.CHECKPOINT_CSV, index=False)
    print("\n✅ All covariates fetched successfully!")
    print(f"📁 Output saved to: {Config.CHECKPOINT_CSV}")

if __name__ == "__main__":
    main()