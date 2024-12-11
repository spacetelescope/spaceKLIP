import functools
import os
from csv import reader
import numpy as np
import astropy
from datetime import datetime, timedelta, timezone
from requests import Session
import spaceKLIP.utils as utils
def extract_oss_event_msgs_for_visit(eventlog, 
                                     selected_visit_id, 
                                     ta_only=False, 
                                     verbose=False, 
                                     return_text=True):
    """
    Extract OSS event messages for a specific visit ID from the event log.

    Parameters
    ----------
    eventlog : list
        List of OSS event log entries.
    selected_visit_id : str
        The visit ID of the observation to filter messages for. The format should be either:
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

    selected_visit_id = utils.get_visitid(selected_visit_id)  # Standardize format to 'VPPPPPOOOVVV'.
    
    # Initialize state variables.
    in_selected_visit = False  # Tracks whether the current log entry belongs to the selected visit.
    in_ta = False  # Tracks whether the current log entry belongs to a TA section.
    messages = []
    vid = ''  # Visit ID.

    if verbose:
        print(f"Searching entries for visit: {selected_visit_id}")

    # Iterate through event log rows.
    for row in eventlog:
        msg, time = row['Message'], row['Time']

        # Collect messages for a selected visit if not restricted to TA messages only.
        if in_selected_visit and (not ta_only or in_ta):
            if verbose:
                print(time[:22], "\t", msg)
            if return_text:
                messages.append(f"{time[:22]}\t{msg}")

        # Find when the visit started.
        if msg.startswith('VISIT ') and msg.endswith('STARTED'):
            vid = msg.split()[1]
            vstart = 'T'.join(time.split())[:-3]
            if vid == selected_visit_id:
                if verbose:
                    print(f"VISIT {selected_visit_id} START FOUND at {vstart}")
                in_selected_visit = True
                if ta_only and verbose:
                    print("Only displaying TARGET ACQUISITION RESULTS:")

        # Find when the visit ended.
        elif msg.startswith('VISIT ') and msg.endswith('ENDED') and in_selected_visit:
            assert vid == msg.split()[1]  # Confirm the visit ID is the same as in start.
            assert selected_visit_id  == msg.split()[1]
            vend = 'T'.join(time.split())[:-3]
            if verbose:
                print(f"VISIT {selected_visit_id} END FOUND at {vend}")
            in_selected_visit = False

        # Check for any script termination errors.
        elif in_selected_visit and msg.startswith(f'Script terminated: {selected_visit_id}') and 'ERROR' in msg:
            in_selected_visit = False
            if verbose:
                print(f"Script terminated with error for visit {selected_visit_id}.")
                
        # This string is used to mark the start and end of TA sections.
        elif in_selected_visit and msg.startswith('*'):
            in_ta = not in_ta

    # Return extracted messages.
    return messages if return_text else None


def extract_oss_TA_centroids(eventlog,
                             selected_visit_id):
    """
    Return the TA centroid values from OSS.
    Note, pretty sure these values from OSS are 1-based pixel coordinates - to be confirmed!
    
    Parameters
    ----------
    eventlog : list
        List of OSS event log entries.
    selected_visit_id : str
        The visit ID of the observation to filter messages for. The format should be either:
        - 'VPPPPPOOOVVV' (e.g., V01234001001)
        - 'PPPP:O:V' (e.g., 1234:0:1).

    Returns
    -------
    tuple
        A tuple (X, Y) of centroid coordinates (assumed 1-based indexing).
    """
    
    # Retrieve TA-related messages for the selected visit.
    msgs = extract_oss_event_msgs_for_visit(eventlog, selected_visit_id,
                                            ta_only=True, verbose=False,
                                            return_text=True)
    
    # Parse messages to find the centroid coordinates.
    for msg in msgs:
        if "detector coord" in msg.split('\t')[1]:
            date_time = " ".join(msg.split()[:2])

            from datetime import datetime
            timestamp = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f")

            print("Extracted date and time:", date_time)
            print("Datetime object:", timestamp)
            return tuple(float(p.strip('(),')) for p in msg.split()[-2:])
    raise RuntimeError(f"Could not parse TA centroid coordinates in visit log for {selected_visit_id}")

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
                         mast_api_token=mast_api_token, verbose=verbose, return_as_table=False)
   
    return parse_eventlog_to_table(lines, label='Message') if return_as_table else lines

@functools.lru_cache
def get_mnemonic(mnemonic, 
                 startdate='2022-02-01', 
                 enddate=None, 
                 mast_api_token=None, 
                 verbose=False, 
                 return_as_table=True, 
                 change_only=True):
    """
    Retrieve a single mnemonic time series from the JWST Engineering database (EDB).

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
    change_only : bool, optional
        If True, filters table to include only rows with changed values.

    Returns
    -------
    astropy.table.Table or list
        A table of mnemonic data or raw CSV lines.
    """
    
    # Configuration.
    # Base URL for downloading files from the EDB.
    base_url = 'https://mast.stsci.edu/jwst/api/v0.1/Download/file?uri=mast:jwstedb'
    mastfmt = '%Y%m%dT%H%M%S'  # Required format for timestamps in the API query.
    tz_utc = timezone(timedelta(hours=0))  # UTC timezone.
    millisec = timedelta(milliseconds=1)  # For precise millisecond-based time calculations.
    colhead = 'theTime'  # Timestamp column in the data.

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
            "MAST API token is not defined. Set the MAST_API_TOKEN environment variable "
            "or provide the mast_api_token parameter to access proprietary data."
        )

    # Handle dates as astropy.Time, datetime, or strings.
    start = startdate if isinstance(startdate, astropy.time.Time) else datetime.fromisoformat(f'{startdate}+00:00')
    end = enddate if isinstance(startdate, astropy.time.Time) else datetime.fromisoformat(f'{enddate}+23:59:59')

    # Fetch event messages from MAST EDB (logs FOS EDB).
    startstr = start.strftime(mastfmt)
    endstr = end.strftime(mastfmt)
    filename = f'{mnemonic}-{startstr}-{endstr}.csv'
    url = f'{base_url}/{filename}'  # Construct API URL.
    
    if verbose:
        print(f"Retrieving mnemonic {mnemonic} data from: {url}")

    response = session.get(url)
    if response.status_code == 401:
        exit('HTTPError 401 - Check your MAST token and EDB authorization. May need to refresh your token if it expired.')
    response.raise_for_status()
    lines = response.content.decode('utf-8').splitlines()

    # Process response.
    if return_as_table:
        table = parse_eventlog_to_table(lines, label=mnemonic)
        return mnemonic_get_changes(table) if change_only else table
    return lines

def parse_eventlog_to_table(eventlog,
                            label="Value"):
    """
    Parse an eventlog as returned from the EngDB to an astropy table, for ease of use.

    Parameters
    ----------
    eventlog : iterable
        The raw event log data as a list of lines or a CSV-like object, where each 
        row contains [timestamp, MJD, value/message].
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

