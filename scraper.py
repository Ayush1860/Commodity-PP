"""
scraper.py — Agmarknet Data Extraction Module.

Dual-mode scraper: REST API (primary) + Selenium fallback.
Extracts daily commodity price data from the Agmarknet 2.0 portal.
"""

import os
import time
import json
import logging
from typing import Optional
from datetime import datetime

import requests
import pandas as pd

import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class AgmarknetScraper:
    """
    Scrapes commodity price data from the Agmarknet portal.

    Primary method: REST API at api.agmarknet.gov.in/v1/
    Fallback method: Selenium-based browser automation.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(config.REQUEST_HEADERS)
        self._filters_cache: Optional[dict] = None

    # ───────────────────────── Filter Lookup ─────────────────────────────

    def get_filters(self) -> dict:
        """
        Fetch and cache the filter lookup data from the Agmarknet 2.0 API.

        The API returns a JSON with structure:
        {"status": true, "data": {
            "cmdt_data": [{"cmdt_id": ..., "cmdt_name": ..., "cmdt_group_id": ...}, ...],
            "state_data": [{"state_id": ..., "state_name": ...}, ...],
            "type_data": [{"id": ..., "type": ...}, ...],
            "grade_data": [...],
            "market_data": [...],
            ...}}
        """
        if self._filters_cache is not None:
            return self._filters_cache

        logger.info("Fetching filter mappings from Agmarknet API...")
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                resp = self.session.get(
                    config.FILTERS_ENDPOINT,
                    timeout=config.REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                payload = resp.json()
                # The actual data is nested under "data" key
                self._filters_cache = payload.get("data", payload)
                logger.info("Filter mappings fetched successfully.")
                return self._filters_cache
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt}/{config.MAX_RETRIES} failed: {e}")
                if attempt < config.MAX_RETRIES:
                    time.sleep(config.RETRY_DELAY)

        raise ConnectionError(
            "Failed to fetch filter mappings from Agmarknet API after "
            f"{config.MAX_RETRIES} attempts."
        )

    # Mapping from logical filter type to actual API field names
    _FILTER_FIELD_MAP = {
        "commodity": {"data_key": "cmdt_data", "name_field": "cmdt_name", "id_field": "cmdt_id"},
        "state":     {"data_key": "state_data", "name_field": "state_name", "id_field": "state_id"},
        "market":    {"data_key": "market_data", "name_field": "mkt_name", "id_field": "id"},
    }

    def _resolve_id(self, filter_type: str, name: str, filters: dict) -> Optional[int]:
        """
        Resolve a human-readable name to its API ID.

        Args:
            filter_type: One of 'commodity', 'state', 'market'.
            name: Human-readable name to look up (case-insensitive).
            filters: The filter dictionary from get_filters().

        Returns:
            Integer ID or None if not found.
        """
        mapping = self._FILTER_FIELD_MAP.get(filter_type)
        if not mapping:
            logger.warning(f"Unknown filter type: '{filter_type}'")
            return None

        items = filters.get(mapping["data_key"], [])
        name_lower = name.strip().lower()
        name_field = mapping["name_field"]
        id_field = mapping["id_field"]

        for item in items:
            item_name = item.get(name_field, "")
            if isinstance(item_name, str) and item_name.strip().lower() == name_lower:
                return int(item[id_field])

        logger.warning(f"Could not resolve '{name}' in '{mapping['data_key']}'.")
        return None

    def _resolve_commodity_group_id(self, commodity_id: int, filters: dict) -> Optional[int]:
        """Look up the commodity group ID for a given commodity ID."""
        for item in filters.get("cmdt_data", []):
            if item.get("cmdt_id") == commodity_id:
                return item.get("cmdt_group_id")
        return None

    def list_commodities(self, group_id: Optional[int] = None) -> list[dict]:
        """List available commodities, optionally filtered by group ID."""
        filters = self.get_filters()
        commodities = filters.get("cmdt_data", [])
        if group_id is not None:
            commodities = [
                c for c in commodities
                if c.get("cmdt_group_id") == group_id
            ]
        return commodities

    def list_states(self) -> list[dict]:
        """List available states."""
        filters = self.get_filters()
        return filters.get("state_data", [])

    # ───────────────────────── REST API Fetching ─────────────────────────

    def fetch_data_api(
        self,
        commodity_name: str,
        state: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        commodity_group: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch price data using the Agmarknet 2.0 REST API.

        Args:
            commodity_name: Name of the commodity (e.g., "Wheat").
            state: State name (e.g., "Madhya Pradesh").
            from_date: Start date string "YYYY-MM-DD". Defaults to 1 year ago.
            to_date: End date string "YYYY-MM-DD". Defaults to today.
            commodity_group: Unused (group is auto-resolved from commodity).

        Returns:
            DataFrame with columns matching the output schema.
        """
        from_date = from_date or config.DEFAULT_FROM_DATE
        to_date = to_date or config.DEFAULT_TO_DATE

        filters = self.get_filters()

        # ── Resolve IDs ──────────────────────────────────────────────────
        commodity_id = self._resolve_id("commodity", commodity_name, filters)
        state_id = self._resolve_id("state", state, filters)

        if commodity_id is None:
            raise ValueError(
                f"Commodity '{commodity_name}' not found in Agmarknet filters. "
                f"Use list_commodities() to see available options."
            )
        if state_id is None:
            raise ValueError(
                f"State '{state}' not found in Agmarknet filters. "
                f"Use list_states() to see available options."
            )

        # Auto-resolve commodity group from commodity
        group_id = self._resolve_commodity_group_id(commodity_id, filters)

        logger.info(
            f"Fetching data: commodity={commodity_name}(ID:{commodity_id}), "
            f"state={state}(ID:{state_id}), group={group_id}, "
            f"range={from_date} to {to_date}"
        )

        # ── Paginated Fetch ──────────────────────────────────────────────
        all_records = []
        page = 1

        while True:
            params = {
                "from_date": from_date,
                "to_date": to_date,
                "data_type": config.DEFAULT_DATA_TYPE,
                "commodity": commodity_id,
                "state": f"[{state_id}]",
                "district": f"[{config.DEFAULT_DISTRICT_ID}]",
                "market": f"[{config.DEFAULT_MARKET_ID}]",
                "grade": f"[{config.DEFAULT_GRADE_ID}]",
                "variety": f"[{config.DEFAULT_VARIETY_ID}]",
                "limit": config.PAGE_SIZE,
                "page": page,
            }
            if group_id is not None:
                params["group"] = group_id

            for attempt in range(1, config.MAX_RETRIES + 1):
                try:
                    resp = self.session.get(
                        config.REPORT_ENDPOINT,
                        params=params,
                        timeout=config.REQUEST_TIMEOUT,
                    )
                    # 404 means no more pages — treat as end of data
                    if resp.status_code == 404:
                        logger.info(f"Page {page}: 404 — no more data available.")
                        payload = None
                        break
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except requests.RequestException as e:
                    logger.warning(
                        f"API request page {page}, attempt {attempt}: {e}"
                    )
                    if attempt < config.MAX_RETRIES:
                        time.sleep(config.RETRY_DELAY)
                    else:
                        raise ConnectionError(
                            f"Failed to fetch page {page} after "
                            f"{config.MAX_RETRIES} retries."
                        )

            # If we got a 404, stop pagination
            if payload is None:
                break

            # ── Parse Response ───────────────────────────────────────────
            # API returns: {"status": true, "data": {"data": [...records...], "pagination": [{"total_count": N, ...}]}}
            data_body = payload.get("data", payload)
            if isinstance(data_body, dict):
                # Records are in data_body["data"], NOT data_body["records"]
                records = data_body.get("data", data_body.get("records", data_body.get("results", [])))
                # Pagination info is in data_body["pagination"][0]
                pagination = data_body.get("pagination", [])
                if isinstance(pagination, list) and pagination:
                    total = pagination[0].get("total_count")
                elif isinstance(pagination, dict):
                    total = pagination.get("total_count", pagination.get("total"))
                else:
                    total = None
            elif isinstance(data_body, list):
                records = data_body
                total = None
            else:
                records = []
                total = None

            if not records:
                logger.info(f"No more records on page {page}. Fetch complete.")
                break

            all_records.extend(records)
            logger.info(f"Page {page}: fetched {len(records)} records "
                        f"(total: {len(all_records)})")

            # Check for pagination end
            if total is not None and len(all_records) >= total:
                break

            page += 1
            time.sleep(0.5)  # polite delay between pages

        if not all_records:
            logger.warning("No records found for the given query.")
            return pd.DataFrame()

        # ── Normalize to DataFrame ───────────────────────────────────────
        df = pd.DataFrame(all_records)
        df = self._standardize_columns(df)

        logger.info(f"Total records fetched: {len(df)}")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw API column names to the standard output schema:
        Date, Market, Commodity, Variety, Arrivals_Tonnes,
        Min_Price, Max_Price, Modal_Price
        """
        # Build a mapping from possible raw names to standard names
        column_map = {}
        raw_cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}

        standard_mappings = {
            "Date": ["arrival_date", "date", "report_date", "price_date",
                      "reported_date"],
            "Market": ["market_name", "market", "apmc_market", "mandi",
                        "market_center"],
            "Commodity": ["commodity_name", "commodity"],
            "Variety": ["variety_name", "variety", "grade", "variety_grade"],
            "Arrivals_Tonnes": ["arrivals", "arrivals_tonnes", "arrival",
                                 "quantity", "arrival_qty", "arrivals_(tonnes)"],
            "Min_Price": ["min_price", "minimum_price", "min"],
            "Max_Price": ["max_price", "maximum_price", "max"],
            "Modal_Price": ["model_price", "modal_price", "modal"],
            "State": ["state_name", "state"],
            "District": ["district_name", "district"],
        }

        for std_name, possible_names in standard_mappings.items():
            for pname in possible_names:
                if pname in raw_cols_lower:
                    column_map[raw_cols_lower[pname]] = std_name
                    break

        df = df.rename(columns=column_map)

        # Parse date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

        # Convert price and quantity columns to numeric
        for col in ["Min_Price", "Max_Price", "Modal_Price", "Arrivals_Tonnes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by date
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        return df

    # ───────────────────────── Selenium Fallback ─────────────────────────

    def fetch_data_selenium(
        self,
        commodity_name: str,
        state: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fallback scraper using Selenium to interact with the Agmarknet portal.

        Uses headless Chrome via webdriver-manager for automatic driver setup.
        Navigates the portal's React UI, fills dropdowns, and scrapes results.
        """
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException
        from webdriver_manager.chrome import ChromeDriverManager

        from_date = from_date or config.DEFAULT_FROM_DATE
        to_date = to_date or config.DEFAULT_TO_DATE

        logger.info(
            f"[Selenium] Scraping: commodity={commodity_name}, state={state}, "
            f"range={from_date} to {to_date}"
        )

        # ── Browser Setup ────────────────────────────────────────────────
        chrome_options = Options()
        if config.SELENIUM_HEADLESS:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            f"user-agent={config.REQUEST_HEADERS['User-Agent']}"
        )

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options,
        )
        driver.implicitly_wait(config.SELENIUM_IMPLICIT_WAIT)
        driver.set_page_load_timeout(config.SELENIUM_PAGE_LOAD_TIMEOUT)

        all_data = []

        try:
            # ── Navigate to Price Report Page ────────────────────────────
            driver.get(config.PRICE_REPORT_URL)
            wait = WebDriverWait(driver, 15)

            # Wait for the main filter form to render
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "select, [class*='dropdown'], [class*='select']")
            ))
            time.sleep(2)  # allow React hydration

            # ── Helper: interact with searchable dropdown ────────────────
            def select_dropdown_option(dropdown_label: str, option_text: str):
                """Click a dropdown trigger, search, and select an option."""
                # Find dropdown containers by label text
                dropdowns = driver.find_elements(
                    By.XPATH,
                    f"//label[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                    f"'abcdefghijklmnopqrstuvwxyz'),'{dropdown_label.lower()}')]"
                    f"/following-sibling::*[1] | "
                    f"//div[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                    f"'abcdefghijklmnopqrstuvwxyz'),'{dropdown_label.lower()}')]"
                    f"/following-sibling::*[1]"
                )

                if not dropdowns:
                    # Fallback: find by placeholder or aria-label
                    dropdowns = driver.find_elements(
                        By.XPATH,
                        f"//*[contains(@placeholder,'{dropdown_label}') or "
                        f"contains(@aria-label,'{dropdown_label}')]"
                    )

                if dropdowns:
                    dropdown = dropdowns[0]
                    dropdown.click()
                    time.sleep(0.5)

                    # Type into search if available
                    search_inputs = driver.find_elements(
                        By.CSS_SELECTOR,
                        "input[type='search'], input[class*='search']"
                    )
                    if search_inputs:
                        search_input = search_inputs[-1]  # most recently appeared
                        search_input.clear()
                        search_input.send_keys(option_text)
                        time.sleep(1)

                    # Click matching option
                    options = driver.find_elements(
                        By.XPATH,
                        f"//*[contains(@class,'option') or contains(@class,'item')]"
                        f"[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                        f"'abcdefghijklmnopqrstuvwxyz'),'{option_text.lower()}')]"
                    )
                    if options:
                        options[0].click()
                        time.sleep(0.5)
                    else:
                        logger.warning(
                            f"Option '{option_text}' not found in "
                            f"'{dropdown_label}' dropdown."
                        )
                else:
                    logger.warning(f"Dropdown '{dropdown_label}' not found on page.")

            # ── Fill Filters ─────────────────────────────────────────────
            select_dropdown_option("Commodity", commodity_name)
            select_dropdown_option("State", state)

            # Set date fields
            date_inputs = driver.find_elements(
                By.CSS_SELECTOR, "input[type='date'], input[class*='date']"
            )
            if len(date_inputs) >= 2:
                # From date
                date_inputs[0].clear()
                date_inputs[0].send_keys(from_date)
                # To date
                date_inputs[1].clear()
                date_inputs[1].send_keys(to_date)

            time.sleep(1)

            # ── Submit / Go ──────────────────────────────────────────────
            go_buttons = driver.find_elements(
                By.XPATH,
                "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                "'abcdefghijklmnopqrstuvwxyz'),'go') or "
                "contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                "'abcdefghijklmnopqrstuvwxyz'),'submit') or "
                "contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                "'abcdefghijklmnopqrstuvwxyz'),'search')]"
            )
            if go_buttons:
                go_buttons[0].click()
                time.sleep(3)

            # ── Scrape Results Table ─────────────────────────────────────
            try:
                wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "table, [class*='table']")
                ))
            except TimeoutException:
                logger.warning("[Selenium] No data table found after submission.")
                return pd.DataFrame()

            # Extract table data
            tables = driver.find_elements(By.TAG_NAME, "table")
            if tables:
                # Parse the first data table found
                headers = []
                header_cells = tables[0].find_elements(By.TAG_NAME, "th")
                for th in header_cells:
                    headers.append(th.text.strip())

                rows = tables[0].find_elements(By.TAG_NAME, "tr")
                for row in rows[1:]:  # skip header row
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if cells:
                        row_data = {
                            headers[i]: cells[i].text.strip()
                            for i in range(min(len(headers), len(cells)))
                        }
                        all_data.append(row_data)

                # Check for pagination (Next button)
                while True:
                    next_btns = driver.find_elements(
                        By.XPATH,
                        "//button[contains(text(),'Next') or "
                        "contains(@aria-label,'Next')]"
                    )
                    if next_btns and next_btns[0].is_enabled():
                        next_btns[0].click()
                        time.sleep(2)
                        rows = tables[0].find_elements(By.TAG_NAME, "tr")
                        for row in rows[1:]:
                            cells = row.find_elements(By.TAG_NAME, "td")
                            if cells:
                                row_data = {
                                    headers[i]: cells[i].text.strip()
                                    for i in range(min(len(headers), len(cells)))
                                }
                                all_data.append(row_data)
                    else:
                        break

        except Exception as e:
            logger.error(f"[Selenium] Error during scraping: {e}")
        finally:
            driver.quit()

        if not all_data:
            logger.warning("[Selenium] No data scraped.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = self._standardize_columns(df)
        logger.info(f"[Selenium] Total records scraped: {len(df)}")
        return df

    # ───────────────────────── Unified Fetch ─────────────────────────────

    def fetch_data(
        self,
        commodity_name: str,
        state: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        commodity_group: Optional[str] = None,
        force_selenium: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch commodity price data with automatic fallback.

        Tries REST API first; falls back to Selenium if API fails.

        Args:
            commodity_name: Commodity to query.
            state: State to query.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).
            commodity_group: Optional group filter.
            force_selenium: If True, skip API and use Selenium directly.

        Returns:
            DataFrame with standardized columns.
        """
        if not force_selenium:
            try:
                logger.info("Attempting REST API fetch...")
                df = self.fetch_data_api(
                    commodity_name=commodity_name,
                    state=state,
                    from_date=from_date,
                    to_date=to_date,
                    commodity_group=commodity_group,
                )
                if not df.empty:
                    return df
                logger.warning("API returned empty data. Falling back to Selenium.")
            except Exception as e:
                logger.warning(f"API fetch failed: {e}. Falling back to Selenium.")

        logger.info("Using Selenium fallback...")
        return self.fetch_data_selenium(
            commodity_name=commodity_name,
            state=state,
            from_date=from_date,
            to_date=to_date,
        )

    # ───────────────────────── CSV Export ─────────────────────────────────

    def save_to_csv(
        self,
        df: pd.DataFrame,
        commodity_name: str,
        state: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> str:
        """
        Save DataFrame to CSV in the data directory.

        Returns:
            Absolute path to the saved CSV.
        """
        from_date = from_date or config.DEFAULT_FROM_DATE
        to_date = to_date or config.DEFAULT_TO_DATE

        # Sanitize names for filename
        safe_commodity = commodity_name.strip().replace(" ", "_").lower()
        safe_state = state.strip().replace(" ", "_").lower()
        filename = f"{safe_commodity}_{safe_state}_{from_date}_to_{to_date}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)

        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to: {filepath}")
        return filepath


# ───────────────────────── CLI Entry Point ───────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape commodity prices from Agmarknet."
    )
    parser.add_argument("--commodity", required=True, help="Commodity name (e.g., Wheat)")
    parser.add_argument("--state", required=True, help="State name (e.g., Madhya Pradesh)")
    parser.add_argument("--from-date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--group", default=None, help="Commodity group (e.g., Cereals)")
    parser.add_argument("--selenium", action="store_true", help="Force Selenium scraping")
    parser.add_argument("--list-commodities", action="store_true",
                        help="List available commodities and exit")
    parser.add_argument("--list-states", action="store_true",
                        help="List available states and exit")

    args = parser.parse_args()
    scraper = AgmarknetScraper()

    if args.list_commodities:
        commodities = scraper.list_commodities()
        print(f"\nAvailable Commodities ({len(commodities)}):")
        for c in commodities:
            print(f"  [{c.get('cmdt_id', '?')}] {c.get('cmdt_name', 'Unknown')}")
    elif args.list_states:
        states = scraper.list_states()
        print(f"\nAvailable States ({len(states)}):")
        for s in states:
            print(f"  [{s.get('state_id', '?')}] {s.get('state_name', 'Unknown')}")
    else:
        data = scraper.fetch_data(
            commodity_name=args.commodity,
            state=args.state,
            from_date=args.from_date,
            to_date=args.to_date,
            commodity_group=args.group,
            force_selenium=args.selenium,
        )

        if data.empty:
            print("No data fetched. Please check your parameters.")
        else:
            csv_path = scraper.save_to_csv(
                data, args.commodity, args.state, args.from_date, args.to_date,
            )
            print(f"\n✓ Fetched {len(data)} records.")
            print(f"  Saved to: {csv_path}")
            print(f"\nSample:\n{data.head()}")
