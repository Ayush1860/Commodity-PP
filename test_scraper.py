"""Quick test of the Agmarknet scraper for vegetable data."""

import os
import logging
import sys

# Write output to file in the project directory
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output.txt")
outfile = open(output_path, "w", encoding="utf-8")
sys.stdout = outfile
sys.stderr = outfile

logging.basicConfig(level=logging.INFO, stream=outfile)

from scraper import AgmarknetScraper

s = AgmarknetScraper()

# Test 1: Filter resolution
filters = s.get_filters()
print("Filter keys:", list(filters.keys()))

# Test 2: ID resolution for a vegetable
cid = s._resolve_id("commodity", "Tomato", filters)
sid = s._resolve_id("state", "Maharashtra", filters)
print(f"Tomato ID: {cid}, Maharashtra ID: {sid}")

# Test 3: Fetch vegetable data
df = s.fetch_data("Tomato", "Maharashtra", from_date="2026-03-01", to_date="2026-03-11")
print(f"\nRESULT: {len(df)} rows fetched")
print(f"Columns: {df.columns.tolist()}")
if not df.empty:
    print(df.head(3).to_string())
else:
    print("EMPTY DataFrame returned!")

outfile.close()
