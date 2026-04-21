"""Quick test of the fixed Agmarknet scraper — writes output to file."""
import logging
import sys

# Write output to file
outfile = open(r"d:\CommodityPP\test_output.txt", "w", encoding="utf-8")
sys.stdout = outfile
sys.stderr = outfile

logging.basicConfig(level=logging.INFO, stream=outfile)

from scraper import AgmarknetScraper

s = AgmarknetScraper()

# Test 1: Filter resolution
filters = s.get_filters()
print("Filter keys:", list(filters.keys()))

# Test 2: ID resolution
cid = s._resolve_id("commodity", "Wheat", filters)
sid = s._resolve_id("state", "Madhya Pradesh", filters)
print(f"Wheat ID: {cid}, Madhya Pradesh ID: {sid}")

# Test 3: Fetch data
df = s.fetch_data("Wheat", "Madhya Pradesh", from_date="2026-03-01", to_date="2026-03-11")
print(f"\nRESULT: {len(df)} rows fetched")
print(f"Columns: {df.columns.tolist()}")
if not df.empty:
    print(df.head(3).to_string())
else:
    print("EMPTY DataFrame returned!")

outfile.close()
