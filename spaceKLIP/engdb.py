import functools
import os
from csv import reader
import numpy as np
import astropy
import warnings
from datetime import datetime, timedelta, timezone
from requests import Session
import spaceKLIP.utils as utils


def extract_oss_event_msgs_for_visit(eventlog,
                                     visit_id,
                                     ta_only=False,
                                     verbose=False,
                                     return_text=True):
    """
    Extract OSS event messages for a specific visit ID from the event log.

    Parameters
    ----------
    eventlog : list
        List of OSS event log entries.
    visit_id : str
        The visit ID of the observation to filter messages for.
        The format should be either:
            - 'VPPPPPOOOVVV' (e.g., V01234001001)
            - 'PPPP:O:V' (e.g., 1234:0:1).
    ta_only : bool, optional
        If True, only include messages related to Target Acquisition (TA).
    verbose : bool, optional
        If True, displays detailed progress messages and logs during execution.
    return_text : bool, optional
        If True, return the messages as a list of strings.

    Returns
    -------
    list of str
        Extracted messages with timestamps if `return_text` is True.
    """
    # Standardize format to 'VPPPPPOOOVVV'.
    visit_id = utils.get_visitid(visit_id)

    # Initialize state variables.
    in_visit = False  # Tracks whether the log entry belongs to visit.
    in_ta = False  # Tracks whether the log entry belongs to a TA section.
    messages = []
    vid = ''  # Visit ID.

    if verbose:
        print(f"Searching entries for visit: {visit_id}")

    # Iterate through event log rows.
    for row in eventlog:
        msg, time = row['Message'], row['Time']
        timestamp = time[:22]  # Format timestamp for printing.

        # Find when the visit started.
        if msg.startswith("VISIT ") and msg.endswith("STARTED"):
            vid = msg.split()[1]
            if vid == visit_id:
                in_visit = True
                in_ta = False  # Reset TA flag.
                if verbose:
                    print(f"VISIT {visit_id} "
                          f"START FOUND at {timestamp}")
                    if ta_only:
                        print("Only displaying TARGET ACQUISITION RESULTS.")
            continue

        # Find when the visit ended.
        if msg.startswith("VISIT ") and msg.endswith("ENDED") and in_visit:
            in_visit = False
            in_ta = False  # Ensure TA tracking stops.
            if verbose:
                print(f"VISIT {visit_id} END FOUND at {timestamp}")
            continue

        # Find the target acquisition sections.
        if in_visit and "TARGET LOCATE SUMMARY" in msg:
            in_ta = True  # Enter TA mode.
            if verbose:
                print(f"{timestamp}\t{msg}")
            if return_text:
                messages.append(f"{timestamp}\t{msg}")
            continue

        # Ensure TA mode exits properly (detect when TA section ends).
        if in_visit and in_ta and "*" in msg:
            in_ta = False  # Exit TA mode.
            continue

        # Collect messages for visit if not restricted to TA messages only.
        if in_visit and (not ta_only or in_ta):
            if verbose:
                print(f"{timestamp}\t{msg}")
            if return_text:
                messages.append(f"{timestamp}\t{msg}")

        # Check for any script termination errors.
        if (in_visit and msg.startswith(f"Script terminated: {visit_id}") and
                'ERROR' in msg):
            in_visit = False
            if verbose:
                print(f"Script terminated with error for visit {visit_id}.")

    # Return extracted messages.
    return messages if return_text else None


def extract_oss_TA_centroids(eventlog,
                             selected_visit_id):
    """
    Return the TA centroid values from OSS.

    Parameters
    ----------
    eventlog : list
        List of OSS event log entries.
    selected_visit_id : str
        The visit ID of the observation to filter messages for.
        The format should be either:
            - 'VPPPPPOOOVVV' (e.g., V01234001001)
            - 'PPPP:O:V' (e.g., 1234:0:1).

    Returns
    -------
    list of tuples
        A list of tuples [(X1, Y1), (X2, Y2), ...], where each tuple contains
        the centroid coordinates detected during the TA process for the
        selected visit (assumed 1-based indexing).
    """

    # Retrieve TA-related messages for the selected visit.
    msgs = extract_oss_event_msgs_for_visit(eventlog, selected_visit_id,
                                            ta_only=True, verbose=False,
                                            return_text=True)

    # Parse messages to find all centroid coordinates.
    centroids = []  # Store multiple centroid coordinates.
    timestamps = []  # Store corresponding timestamps.

    for msg in msgs:
        if "detector coord" in msg.split('\t')[1]:
            date_time = " ".join(msg.split()[:2])

            from datetime import datetime
            timestamp = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f")
            timestamps.append(timestamp)

            # Extract centroid coordinates.
            coords = tuple(float(p.strip('(),')) for p in msg.split()[-2:])
            centroids.append(coords)

            # Print extracted details.
            print("\nExtracted date and time:", date_time)
            print("Extracted Centroid Coordinates:", coords)

    # If no centroid coordinates are found, raise an error.
    if not centroids:
        raise RuntimeError((f"Could not parse TA centroid coordinates "
                            f"in visit log for {selected_visit_id}"))

    # Return the full list of extracted centroid coordinates and timestamps.
    return centroids


@functools.lru_cache
def get_ictm_event_log(startdate='2022-02-01',
                       enddate=None,
                       mast_api_token=None,
                       verbose=False,
                       return_as_table=True):
    """
    Retrieve the ICTM_EVENT_MSG event log within a specified date range.

    Parameters
    ----------
    startdate : str
        Start date for the query in 'YYYY-MM-DD' format.
    enddate : str, optional
        End date for the query in 'YYYY-MM-DD' format.
    mast_api_token : str, optional
        Token for MAST API authentication. Required to access proprietary data.
    verbose : bool, optional
        If True, displays detailed progress messages and logs during execution.
    return_as_table : bool, optional
        If True, returns the result as a table; otherwise, returns raw lines.

    Returns
    -------
    Table or list
        Parsed table or raw lines from the ICTM_EVENT_MSG event log.
    """

    # Define the mnemonic for the event log.
    mnemonic = 'ICTM_EVENT_MSG'

    # Fetch lines from the event log.
    lines = get_mnemonic(mnemonic, startdate=startdate, enddate=enddate,
                         mast_api_token=mast_api_token, verbose=verbose,
                         return_as_table=False)

    log_table = parse_eventlog_to_table(lines, label='Message')
    return log_table if return_as_table else lines


@functools.lru_cache
def get_mnemonic(mnemonic,
                 startdate='2022-02-01',
                 enddate=None,
                 mast_api_token=None,
                 verbose=False,
                 return_as_table=True):
    """
    Retrieve a single mnemonic time series from the
    JWST Engineering database (EDB).

    Parameters
    ----------
    mnemonic : str
        Identifier for the telemetry parameter.
    startdate : str
        Start date for the query in 'YYYY-MM-DD' format.
    enddate : str, optional
        End date for the query in 'YYYY-MM-DD' format.
    mast_api_token : str, optional
        Token for MAST API authentication. Required to access proprietary data.
    verbose : bool, optional
        If True, displays detailed progress messages and logs during execution.
    return_as_table : bool, optional
        If True, returns the result as a table; otherwise, returns raw lines.

    Returns
    -------
    astropy.table.Table or list
        A table of mnemonic data or raw CSV lines.
    """

    # Configuration.
    # Base URL for downloading files from the EDB.
    base_url = ('https://mast.stsci.edu/jwst/api/v0.1/'
                'Download/file?uri=mast:jwstedb')
    mastfmt = '%Y%m%dT%H%M%S'  # Required format for timestamps in API query.
    tz_utc = timezone(timedelta(hours=0))  # UTC timezone.

    # Set enddate to the current time if it is None.
    tz_utc = timezone.utc
    enddate = enddate or datetime.now(tz=tz_utc).date()

    # Establish MAST session with authorization token.
    session = Session()
    mast_api_token = mast_api_token or os.environ.get('MAST_API_TOKEN')
    if mast_api_token:
        session.headers.update({'Authorization': f'token {mast_api_token}'})
    else:
        warnings.warn(
            "MAST API token is not defined. "
            "Set the MAST_API_TOKEN environment variable or provide the "
            "mast_api_token parameter to access proprietary data."
        )

    # Handle dates as astropy.Time, datetime, or strings.
    start = (startdate if isinstance(startdate, astropy.time.Time)
             else datetime.fromisoformat(f'{startdate}+00:00'))
    end = (enddate if isinstance(startdate, astropy.time.Time)
           else datetime.fromisoformat(f'{enddate}+23:59:59'))

    # Fetch event messages from MAST EDB (logs FOS EDB).
    startstr = start.strftime(mastfmt)
    endstr = end.strftime(mastfmt)
    filename = f'{mnemonic}-{startstr}-{endstr}.csv'
    url = f'{base_url}/{filename}'  # Construct API URL.

    if verbose:
        print(f"Retrieving mnemonic {mnemonic} data from: {url}")

    response = session.get(url)
    if response.status_code == 401:
        exit('HTTPError 401 - Check your MAST token and EDB authorization.'
             'May need to refresh your token if it expired.')
    response.raise_for_status()
    lines = response.content.decode('utf-8').splitlines()

    # Process response.
    if return_as_table:
        table = parse_eventlog_to_table(lines, label=mnemonic)
        return table
    return lines


def parse_eventlog_to_table(eventlog,
                            label="Value"):
    """
    Parse an eventlog as returned from the EngDB to an astropy table.

    Parameters
    ----------
    eventlog : iterable
        The raw event log data as a list of lines or a CSV-like object,
        where each row contains [timestamp, MJD, value/message].
    label : str, optional
        The name for the third column of the table.

    Returns
    -------
    event_table : astropy.table.Table
        A table with columns: "Time", "MJD", and the provided label.
    """

    # Parse and skip the header row.
    rows = list(reader(eventlog, delimiter=",", quotechar='"'))[1:]
    timestr, mjd, messages, *_ = zip(*rows)  # Ignore any extra columns.

    # Convert data types.
    mjd = np.asarray(mjd, dtype=float)
    try:
        messages = np.asarray(messages, dtype=float)
    except ValueError:
        pass  # Leave messages as strings if conversion fails.

    # Assemble into an astropy table.
    event_table = astropy.table.Table([timestr, mjd, messages],
                                      names=["Time", "MJD", label])
    return event_table
