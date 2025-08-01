import logging
# Define logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

from contextlib import contextmanager

@contextmanager
def crds_logging_disabled(highest_level=-3):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    Usage to suppress levels up to Critical:

        >>> with crds_logging_disabled(-3):
        >>>     do_something()

    Parameters
    ----------
    highest_level : int
        For CRDS, logging levels are defined as:
        -3: CRITICAL
        -2: ERROR
        -1: WARNING
         0: INFO
         1: DEBUG
    """
    from crds.core import log as crds_log
    previous_level = crds_log.get_verbose()
    crds_log.set_verbose(highest_level)
    try:
        yield
    finally:
        crds_log.set_verbose(previous_level)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    Usage to suppress levels up to INFO:

        >>> with all_logging_disabled(logging.INFO):
        >>>     do_something()

    Parameters
    ----------
    highest_level : int
        The highest logging level that will be let through. Any
        logging messages at this level or above will be processed
        as normal. The default is CRITICAL, which will allow only
        CRITICAL messages through.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level-1)

    try:
        yield
    finally:
        logging.disable(previous_level)

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""

    import sys, os

    prev_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")

    try:
        with open(os.devnull, "w") as fh:
            sys.stdout = fh
            yield
    finally:
        sys.stdout = prev_stdout # reset old stdout
