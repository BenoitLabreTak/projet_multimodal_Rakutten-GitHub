import datetime

def generate_version() -> str:
    """
    Generate a version string based on the current date and time.
    """
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return dt