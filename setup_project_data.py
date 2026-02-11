import os
import time
import shutil
import zipfile
import requests
import glob
import pandas as pd
from io import StringIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

DIRS = {
    "of": os.path.join(DATA_DIR, "openflights"),
    "caa": os.path.join(DATA_DIR, "caa_uk"),
    "anac": os.path.join(DATA_DIR, "anac_br"),
    "bts": os.path.join(DATA_DIR, "bts_usa"),
    "temp": os.path.join(BASE_DIR, "temp_downloads")
}

BTS_YEAR = "2024"


def ensure_dirs():
    print("Directories creation:")
    for key, path in DIRS.items():
        os.makedirs(path, exist_ok=True)
        print(f"   -> OK: {path}")


def download_simple(url, dest_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    Downloaded: {os.path.basename(dest_path)}")
    except Exception as e:
        print(f"   X Error downloading {url}: {e}")

# ================= 1. OPENFLIGHTS =================


def process_openflights():
    print("\n[1/4] Processing OpenFlights...")

    # --- Airports ---
    url_air = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    cols_air = ["AirportID", "Name", "City", "Country", "IATA", "ICAO", "Latitude",
                "Longitude", "Altitude", "Timezone", "DST", "TzDatabaseTimeZone", "Type", "Source"]
    path_air = os.path.join(DIRS["of"], "airports.csv")

    try:
        r = requests.get(url_air)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), header=None, names=cols_air)
        df.to_csv(path_air, index=False)
        print("    airports.csv saved with header.")
    except Exception as e:
        print(f"    X Error Airports: {e}")

    # --- Routes ---
    url_route = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
    cols_route = ["Airline", "AirlineID", "SourceAirport", "SourceAirportID",
                  "DestAirport", "DestAirportID", "Codeshare", "Stops", "Equipment"]
    path_route = os.path.join(DIRS["of"], "routes.csv")

    try:
        r = requests.get(url_route)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), header=None, names=cols_route)
        df.to_csv(path_route, index=False)
        print("    routes.csv saved with header.")
    except Exception as e:
        print(f"    X Error Routes: {e}")

# ================= 2. CAA UK =================


def process_caa():
    print("\n[2/4] Processing CAA UK...")
    # Direct link extracted from your HTML
    url = "https://www.caa.co.uk/Documents/Download/12040/43d8c177-ce46-40bd-9b0a-ac45f3bdaaec/1654"
    filename = "2024_Annual_Punctuality_Statistics.csv"
    dest = os.path.join(DIRS["caa"], filename)
    download_simple(url, dest)

# ================= 3. ANAC BRAZIL =================


def process_anac():
    print("\n[3/4] Processing ANAC Brazil (VRA 2024)...")

    base_url = "https://sistemas.anac.gov.br/dadosabertos/Voos%20e%20opera%C3%A7%C3%B5es%20a%C3%A9reas/Voo%20Regular%20Ativo%20%28VRA%29/2024"

    # Mapping of folder names on the ANAC server
    months_map = {
        1: "01%20-%20Janeiro", 2: "02%20-%20Fevereiro", 3: "03%20-%20Mar%C3%A7o",
        4: "04%20-%20Abril",   5: "05%20-%20Maio",      6: "06%20-%20Junho",
        7: "07%20-%20Julho",   8: "08%20-%20Agosto",    9: "09%20-%20Setembro",
        10: "10%20-%20Outubro", 11: "11%20-%20Novembro", 12: "12%20-%20Dezembro"
    }

    for m in range(1, 13):
        # Construct URL: .../2024/01%20-%20Janeiro/VRA_20241.csv
        folder = months_map[m]
        filename = f"VRA_2024{m}.csv"
        url = f"{base_url}/{folder}/{filename}"
        dest = os.path.join(DIRS["anac"], filename)

        # Download file
        print(f"   -> Fetching {filename}...", end="")
        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open(dest, 'wb') as f:
                    f.write(r.content)
                print(" OK")
            else:
                print(f" ERR (Status {r.status_code})")
        except Exception as e:
            print(f" ERR: {e}")

# ================= 4. BTS USA (SELENIUM) =================


def process_bts():
    print("\n[4/4] Processing BTS USA (Selenium)...")

    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": DIRS["temp"],
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--headless")  # Uncomment if you don't want to see the browser

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    url_bts = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK&QO_fu146_anzr=b0-gvzr"

    try:
        driver.get(url_bts)
        time.sleep(5)

        for m in range(1, 13):
            str_m = str(m)
            target_csv_name = f"bts_ontime_{BTS_YEAR}_{m:02d}.csv"
            final_path = os.path.join(DIRS["bts"], target_csv_name)

            print(f"    -> Processing {BTS_YEAR}-{m:02d}...", end=" ")

            success = False
            for attempt in range(3):
                try:
                    # 1. Select Year
                    Select(driver.find_element(By.ID, "cboYear")).select_by_value(BTS_YEAR)
                    time.sleep(2)
                    # 2. Select Month
                    Select(driver.find_element(By.ID, "cboPeriod")).select_by_value(str_m)
                    time.sleep(1)
                    # 3. Check Zip
                    chk = driver.find_element(By.ID, "chkDownloadZip")
                    if not chk.is_selected():
                        chk.click()

                    # 4. Download
                    btn = driver.find_element(By.ID, "btnDownload")
                    driver.execute_script("arguments[0].click();", btn)

                    # Wait download (90 sec)
                    timeout = 0
                    while not glob.glob(os.path.join(DIRS["temp"], "*.zip")) and timeout < 90:
                        time.sleep(1)
                        timeout += 1

                    list_of_files = glob.glob(os.path.join(DIRS["temp"], "*.zip"))
                    if not list_of_files:
                        raise Exception("Timeout download ZIP")

                    # 5. Unzip
                    time.sleep(2)
                    latest_zip = max(list_of_files, key=os.path.getctime)

                    with zipfile.ZipFile(latest_zip, 'r') as z:
                        csv_in_zip = [f for f in z.namelist() if f.lower().endswith('.csv')][0]
                        with z.open(csv_in_zip) as source, open(final_path, "wb") as target:
                            shutil.copyfileobj(source, target)

                    print(f" OK (saved as {target_csv_name})")
                    success = True

                    for f in os.listdir(DIRS["temp"]):
                        try:
                            os.remove(os.path.join(DIRS["temp"], f))
                        except:
                            pass

                    break

                except Exception as e:
                    print(f"\n      [Retry {attempt+1}/3] Error: {e}. Reload page...")
                    driver.get(url_bts)
                    time.sleep(5)

                    for f in os.listdir(DIRS["temp"]):
                        try:
                            os.remove(os.path.join(DIRS["temp"], f))
                        except:
                            pass

            if not success:
                print(f"    FAILED to process {BTS_YEAR}-{m:02d} after 3 attempts.")

    finally:
        driver.quit()
        if os.path.exists(DIRS["temp"]):
            shutil.rmtree(DIRS["temp"])


if __name__ == "__main__":
    print(" --- STARTING DATA RESTORE PROCESS ---")
    ensure_dirs()
    process_openflights()
    process_caa()
    process_anac()
    process_bts()
    print("\n\nALL DONE! The 'data/' folder is ready for analysis.")
