import requests
import io
import json
import pandas as pd

from rca_data_tools.qaqc.utils import select_logger


def loadQARTOD(refDes, param, sensorType, logger=select_logger()):
    renameMap = {
        "sea_water_temperature": "seawater_temperature",
        "sea_water_practical_salinity": "practical_salinity",
        "sea_water_pressure": "seawater_pressure",
        "sea_water_density": "density",
        #'ph_seawater':'seawater_ph',
    }

    if param in renameMap:
        param = renameMap[param]

    (site, node, sensor1, sensor2) = refDes.split("-")
    sensor = sensor1 + "-" + sensor2

    # if parameter is oxygen sensor in ctd stream, replace sensor type
    if sensorType == "ctdpf" and "oxygen" in param:
        if "SF" in node:
            sensorType = "dofst"
        else:
            sensorType = "dosta"

    # Load climatology and gross range values

    githubBaseURL = (
        "https://raw.githubusercontent.com/oceanobservatories/qc-lookup/master/qartod/"
    )

    if sensorType == "phsen":
        param = "seawater_ph"

    clim_URL = (
        githubBaseURL + sensorType + "/climatology_tables/" + refDes + "-" + param + ".csv"
    )
    grossRange_URL = (
        githubBaseURL + sensorType + "/" + sensorType + "_qartod_gross_range_test_values.csv"
    )

    download = requests.get(grossRange_URL)
    if download.status_code == 200:
        df_grossRange = pd.read_csv(io.StringIO(download.content.decode("utf-8")))
        qcConfig = df_grossRange.qcConfig[
            (df_grossRange.subsite == site)
            & (df_grossRange.node == node)
            & (df_grossRange.sensor == sensor)
            & (df_grossRange.parameters.str.contains(param))
        ]
        if len(qcConfig) > 0:
            qcConfig_json = qcConfig.values[0].replace("'", '"')
            grossRange_dict = json.loads(qcConfig_json)
        else:
            logger.warning(
                f"error retrieving gross range data for {refDes} {param} {sensorType}"
            )
            grossRange_dict = {}
    else:
        logger.warning(f"error retrieving gross range data for {refDes} {param} {sensorType}")
        grossRange_dict = {}

    download = requests.get(clim_URL)
    if download.status_code == 200:
        df_clim = pd.read_csv(io.StringIO(download.content.decode("utf-8")))
        climRename = {
            "Unnamed: 0": "depth",
            "[1, 1]": "1",
            "[2, 2]": "2",
            "[3, 3]": "3",
            "[4, 4]": "4",
            "[5, 5]": "5",
            "[6, 6]": "6",
            "[7, 7]": "7",
            "[8, 8]": "8",
            "[9, 9]": "9",
            "[10, 10]": "10",
            "[11, 11]": "11",
            "[12, 12]": "12",
        }

        df_clim.rename(columns=climRename, inplace=True)
        clim_dict = df_clim.set_index("depth").to_dict()
    else:
        logger.warning(f"error retrieving climatology data for {refDes} {param} {sensorType}")
        clim_dict = {}

    return (grossRange_dict, clim_dict)


def loadStagedQARTOD(refDes, param, table_type, logger=select_logger()):
    """
    Similar to loadQARTOD, but loads from RCA staging repo and acounts for slight differences
    in table formatting.
    Parameters:
    refDes: str
    param: str
    table_type: tuple (climatology, gross range) (fixed, fixed) or (binned, int) for profiler
    """
    # FIXME this rename map is not helping us avoid 404 errors when looking for tables
    # FIXME might need a special staging rename map? actualy might not need this for staged qartod
    # renameMap = {
    #     "sea_water_temperature": "seawater_temperature",
    #     "sea_water_practical_salinity": "practical_salinity",
    #     "sea_water_pressure": "seawater_pressure",
    #     "sea_water_density": "density",
    #     #'ph_seawater':'seawater_ph',
    # }

    # if param in renameMap:
    #     param = renameMap[param]

    (site, node, sensor1, sensor2) = refDes.split("-")
    sensor = sensor1 + "-" + sensor2

    githubBaseURL = "https://raw.githubusercontent.com/wruef/qartod_staging/refs/heads/main/"
    # ie CE04OSPS-PC01B-4B-PHSENA106-ph_seawater.climatology.csv.fixed
    if table_type[0] is None:
        clim_dict = None
    else:
        clim_URL = (
            githubBaseURL
            + refDes
            + "-"
            + param
            + ".climatology_table.csv."
            + table_type[0]  # clim table type
        )
        # get clim table
        download = requests.get(clim_URL)
        if download.status_code == 200:
            df_clim = pd.read_csv(io.StringIO(download.content.decode("utf-8")))
            climRename = {
                "Unnamed: 0": "depth",
                "[1, 1]": "1",
                "[2, 2]": "2",
                "[3, 3]": "3",
                "[4, 4]": "4",
                "[5, 5]": "5",
                "[6, 6]": "6",
                "[7, 7]": "7",
                "[8, 8]": "8",
                "[9, 9]": "9",
                "[10, 10]": "10",
                "[11, 11]": "11",
                "[12, 12]": "12",
            }

            df_clim.rename(columns=climRename, inplace=True)
            clim_dict = df_clim.set_index("depth").to_dict()
        else:
            raise FileNotFoundError(f"Climatology table not found at {clim_URL}")

    if table_type[1] is None:
        gross_dict = None
    else:
        grossRange_URL = (
            githubBaseURL + refDes + "-" + param + "-gross_range_test_values.csv"
            # + table_type[1] # gross range table type #NOTE wendi dropped this tag in staging
        )
        download = requests.get(grossRange_URL)
        if download.status_code == 200:
            df_grossRange = pd.read_csv(io.StringIO(download.content.decode("utf-8")))
            qcConfig = df_grossRange.qcConfig[
                (df_grossRange.subsite == site)
                & (df_grossRange.node == node)
                & (df_grossRange.sensor == sensor)
                & (df_grossRange.parameters.str.contains(param))
            ]
            if len(qcConfig) > 0:
                qcConfig_json = qcConfig.values[0].replace("'", '"')
            gross_dict = json.loads(qcConfig_json)

        else:
            raise FileNotFoundError(f"Gross range table not found at {grossRange_URL}")

    return clim_dict, gross_dict
