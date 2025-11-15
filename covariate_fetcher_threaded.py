# -*- coding: utf-8 -*-
"""
Climate Covariates Fetcher - OPTIMIZED VERSION
Fetches NDVI, Temperature, and Precipitation data with parallel processing

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
   - GEE_MAX_WORKERS: Number of parallel GEE requests (10 recommended)
   - OM_MAX_WORKERS: Number of parallel Open-Meteo requests (2-5 recommended)
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
import sys
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
    GRID_SIZE_KM = 25  # Desired grid size in kilometers
    
    # Minimum resolutions for each data source (in km)
    MIN_NDVI_RES_KM = 0.25  # MODIS NDVI minimum: 250m = 0.25km
    MIN_TEMP_PRECIP_RES_KM = 25  # Open-Meteo minimum: ~25km
    
    # Calculate effective resolutions
    NDVI_RES_KM = max(GRID_SIZE_KM, MIN_NDVI_RES_KM)
    TEMP_PRECIP_RES_KM = max(GRID_SIZE_KM, MIN_TEMP_PRECIP_RES_KM)
    
    # Convert km to degrees (1 degree ≈ 111 km)
    NDVI_RES_DEG = NDVI_RES_KM / 111.0
    TEMP_PRECIP_RES_DEG = TEMP_PRECIP_RES_KM / 111.0
    
    # ===== PARALLEL PROCESSING CONFIGURATION =====
    # Adjust these based on API rate limits and your internet speed
    GEE_MAX_WORKERS = 10      # Google Earth Engine (can handle more)
    OM_MAX_WORKERS = 3        # Open-Meteo (be conservative to avoid 429 errors)
    
    # ===== BATCH SIZES =====
    GEE_BATCH_SIZE = 200      # Process 200 locations per batch
    OM_BATCH_SIZE = 100       # Smaller batches for stricter API
    
    # ===== FILE PATHS =====
    SRC_CSV = f"{BASE_DIR}/analysis_panel.csv"
    CHECKPOINT_CSV = f"{BASE_DIR}/climate_checkpoint.csv"
    GEE_CACHE = f"{BASE_DIR}/gee_cache.json"
    OM_CACHE = f"{BASE_DIR}/om_cache.json"
    
    # ===== COLUMN NAMES =====
    NDVI_COL = f"ndvi_mean_(0–1)_({NDVI_RES_KM}x{NDVI_RES_KM}km)"
    TEMP_COL = f"temp_mean_(°C)_({TEMP_PRECIP_RES_KM}x{TEMP_PRECIP_RES_KM}km)"
    PRECIP_COL = f"precip_sum_(mm/year)_({TEMP_PRECIP_RES_KM}x{TEMP_PRECIP_RES_KM}km)"
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print(f"\n⚙️  CONFIGURATION")
        print(f"{'='*60}")
        print(f"Grid Size:        {cls.GRID_SIZE_KM} km")
        print(f"\nEffective Resolutions:")
        print(f"  NDVI:           {cls.NDVI_RES_KM} km ({cls.NDVI_RES_DEG:.4f}°)")
        print(f"  Temp/Precip:    {cls.TEMP_PRECIP_RES_KM} km ({cls.TEMP_PRECIP_RES_DEG:.4f}°)")
        print(f"\nParallel Workers:")
        print(f"  GEE (NDVI):     {cls.GEE_MAX_WORKERS} workers")
        print(f"  Open-Meteo:     {cls.OM_MAX_WORKERS} workers")
        print(f"\nBatch Sizes:")
        print(f"  GEE:            {cls.GEE_BATCH_SIZE} locations/batch")
        print(f"  Open-Meteo:     {cls.OM_BATCH_SIZE} locations/batch")
        
        if cls.GRID_SIZE_KM < cls.MIN_NDVI_RES_KM:
            print(f"\n⚠️  Grid size {cls.GRID_SIZE_KM}km < NDVI min ({cls.MIN_NDVI_RES_KM}km)")
            print(f"   Using NDVI minimum: {cls.NDVI_RES_KM}km")
        if cls.GRID_SIZE_KM < cls.MIN_TEMP_PRECIP_RES_KM:
            print(f"⚠️  Grid size {cls.GRID_SIZE_KM}km < Temp/Precip min ({cls.MIN_TEMP_PRECIP_RES_KM}km)")
            print(f"   Using Temp/Precip minimum: {cls.TEMP_PRECIP_RES_KM}km")
        print(f"{'='*60}\n")

# ============================================================
# Utility Functions
# ============================================================

def initialize_environment():
    """Initialize Earth Engine and create necessary directories"""
    os.makedirs(Config.BASE_DIR, exist_ok=True)
    
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
# GEE NDVI Fetcher (Optimized with Threading)
# ============================================================

def round_coord_gee(lat, lon):
    """Round coordinates to NDVI resolution grid"""
    deg = Config.NDVI_RES_DEG
    return (round(lat / deg) * deg, round(lon / deg) * deg)

def fetch_ndvi_single(lat, lon, year, max_retries=3):
    """Fetch NDVI for a single location/year with retry logic"""
    delay = 10
    buffer_size = Config.NDVI_RES_KM * 1000 / 2  # Half grid size in meters
    
    for attempt in range(max_retries):
        try:
            point = ee.Geometry.Point(lon, lat)
            
            ndvi_coll = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select("NDVI")
            )
            
            if ndvi_coll.size().getInfo() > 0:
                ndvi_mean = ndvi_coll.mean().reduceRegion(
                    ee.Reducer.mean(), point.buffer(buffer_size), 250
                ).get("NDVI").getInfo()
                
                if ndvi_mean is not None:
                    return {Config.NDVI_COL: ndvi_mean / 10000}
            
            return {Config.NDVI_COL: None}
            
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                if attempt < max_retries - 1:
                    print(f"⏸️ GEE rate limit - sleeping {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"❌ GEE rate limit exceeded after {max_retries} retries")
                    raise RuntimeError("GEE_RATE_LIMIT")
            else:
                if attempt == max_retries - 1:
                    print(f"⚠️ NDVI fetch failed for {lat},{lon},{year}: {e}")
                    return {Config.NDVI_COL: None}

def process_ndvi(df, cache):
    """Process NDVI with parallel fetching"""
    print("\n🌿 Starting NDVI fetch (Parallel Processing)...")
    
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
        
        for i in tqdm(range(0, len(coords), Config.GEE_BATCH_SIZE), desc=f"Year {year}"):
            batch = coords[i:i+Config.GEE_BATCH_SIZE]
            to_fetch = []
            
            # Check cache first
            for lat, lon in batch:
                cache_key = f"{year},{lat},{lon}"
                mask = (df["year"] == year) & (df["gee_lat"] == lat) & (df["gee_lon"] == lon)
                
                if cache_key in cache:
                    vals = cache[cache_key]
                    for k, v in vals.items():
                        df.loc[mask, k] = v
                else:
                    to_fetch.append((lat, lon))
            
            if not to_fetch:
                continue
            
            # Parallel fetch
            new_data = {}
            try:
                with ThreadPoolExecutor(max_workers=Config.GEE_MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(fetch_ndvi_single, lat, lon, year): (lat, lon)
                        for lat, lon in to_fetch
                    }
                    
                    for fut in as_completed(futures):
                        lat, lon = futures[fut]
                        try:
                            vals = fut.result()
                            new_data[(year, lat, lon)] = vals
                        except RuntimeError as e:
                            if "GEE_RATE_LIMIT" in str(e):
                                print("\n💾 Saving progress before exiting...")
                                df.to_csv(Config.CHECKPOINT_CSV, index=False)
                                save_cache(cache, Config.GEE_CACHE)
                                sys.exit(1)
                        except Exception as e:
                            print(f"⚠️ Failed {lat},{lon},{year}: {e}")
                            new_data[(year, lat, lon)] = {Config.NDVI_COL: None}
            
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user. Saving progress...")
                df.to_csv(Config.CHECKPOINT_CSV, index=False)
                save_cache(cache, Config.GEE_CACHE)
                sys.exit(0)
            
            # Write results to dataframe and cache
            for (y, lat, lon), vals in new_data.items():
                cache_key = f"{y},{lat},{lon}"
                cache[cache_key] = vals
                mask = (df["year"] == y) & (df["gee_lat"] == lat) & (df["gee_lon"] == lon)
                for k, v in vals.items():
                    df.loc[mask, k] = v
            
            # Save progress
            df.to_csv(Config.CHECKPOINT_CSV, index=False)
            save_cache(cache, Config.GEE_CACHE)
            
            time.sleep(2)  # Rate limit safety
    
    return df, cache

# ============================================================
# Open-Meteo Temperature & Precipitation Fetcher (Optimized)
# ============================================================

def round_coord_om(lat, lon):
    """Round coordinates to Temperature/Precipitation resolution grid"""
    deg = Config.TEMP_PRECIP_RES_DEG
    return (round(lat / deg) * deg, round(lon / deg) * deg)

def fetch_temp_precip_single(lat, lon, year, max_retries=3):
    """Fetch temperature and precipitation for a single location/year"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "UTC"
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                if attempt < max_retries - 1:
                    delay = 10 * (2 ** attempt)
                    print(f"⏸️ Open-Meteo rate limit - sleeping {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError("OM_RATE_LIMIT")
            
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
            
            return {
                Config.TEMP_COL: temp_mean,
                Config.PRECIP_COL: precip_sum
            }
            
        except RuntimeError:
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"⚠️ Open-Meteo error for {lat},{lon},{year}: {e}")
                return {Config.TEMP_COL: None, Config.PRECIP_COL: None}
            time.sleep(5)

def process_temp_precip(df, cache):
    """Process temperature and precipitation with parallel fetching"""
    print("\n🌡️ Starting Temperature & Precipitation fetch (Parallel Processing)...")
    
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
        
        for i in tqdm(range(0, len(coords), Config.OM_BATCH_SIZE), desc=f"Year {year}"):
            batch = coords[i:i+Config.OM_BATCH_SIZE]
            to_fetch = []
            
            # Check cache first
            for lat, lon in batch:
                cache_key = f"{year},{lat},{lon}"
                mask = (df["year"] == year) & (df["om_lat"] == lat) & (df["om_lon"] == lon)
                
                if cache_key in cache:
                    vals = cache[cache_key]
                    for k, v in vals.items():
                        df.loc[mask, k] = v
                else:
                    to_fetch.append((lat, lon))
            
            if not to_fetch:
                continue
            
            # Parallel fetch
            new_data = {}
            try:
                with ThreadPoolExecutor(max_workers=Config.OM_MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(fetch_temp_precip_single, lat, lon, year): (lat, lon)
                        for lat, lon in to_fetch
                    }
                    
                    for fut in as_completed(futures):
                        lat, lon = futures[fut]
                        try:
                            vals = fut.result()
                            new_data[(year, lat, lon)] = vals
                        except RuntimeError as e:
                            if "OM_RATE_LIMIT" in str(e):
                                print("\n💾 Saving progress before exiting...")
                                df.to_csv(Config.CHECKPOINT_CSV, index=False)
                                save_cache(cache, Config.OM_CACHE)
                                sys.exit(1)
                        except Exception as e:
                            print(f"⚠️ Failed {lat},{lon},{year}: {e}")
                            new_data[(year, lat, lon)] = {Config.TEMP_COL: None, Config.PRECIP_COL: None}
            
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user. Saving progress...")
                df.to_csv(Config.CHECKPOINT_CSV, index=False)
                save_cache(cache, Config.OM_CACHE)
                sys.exit(0)
            
            # Write results to dataframe and cache
            for (y, lat, lon), vals in new_data.items():
                cache_key = f"{y},{lat},{lon}"
                cache[cache_key] = vals
                mask = (df["year"] == y) & (df["om_lat"] == lat) & (df["om_lon"] == lon)
                for k, v in vals.items():
                    df.loc[mask, k] = v
            
            # Save progress
            df.to_csv(Config.CHECKPOINT_CSV, index=False)
            save_cache(cache, Config.OM_CACHE)
            
            time.sleep(1)  # Rate limit safety
    
    return df, cache

# ============================================================
# Main Execution
# ============================================================

def main():
    """Main execution function"""
    print("🚀 Starting Climate Covariates Fetcher (OPTIMIZED)\n")
    
    # Display configuration
    Config.print_config()
    
    # Initialize
    initialize_environment()
    df = load_or_create_checkpoint()
    
    # Load caches
    gee_cache = load_or_create_cache(Config.GEE_CACHE)
    om_cache = load_or_create_cache(Config.OM_CACHE)
    
    print(f"📊 Loaded {len(df)} total records")
    print(f"💾 GEE Cache: {len(gee_cache)} entries")
    print(f"💾 Open-Meteo Cache: {len(om_cache)} entries")
    
    # Fetch NDVI
    df, gee_cache = process_ndvi(df, gee_cache)
    
    # Fetch Temperature & Precipitation
    df, om_cache = process_temp_precip(df, om_cache)
    
    # Final save
    df.to_csv(Config.CHECKPOINT_CSV, index=False)
    print("\n✅ All covariates fetched successfully!")
    print(f"📁 Output saved to: {Config.CHECKPOINT_CSV}")
    print(f"\n📈 Final Statistics:")
    print(f"   NDVI:         {df[Config.NDVI_COL].notna().sum()}/{len(df)} filled")
    print(f"   Temperature:  {df[Config.TEMP_COL].notna().sum()}/{len(df)} filled")
    print(f"   Precipitation: {df[Config.PRECIP_COL].notna().sum()}/{len(df)} filled")

if __name__ == "__main__":
    main()