def parse_iso_date(date_text):
    """Parse a date in ISO year-month-day format."""
    return date_text.split("-")


def parse_csv_header(header_line):
    """Extract field names from a comma-separated header."""
    return [field.strip() for field in header_line.split(",")]


def validate_email_address(address):
    """Perform a basic check for an email address separator."""
    return "@" in address


def calculate_shipping_fee(weight, base_rate):
    """Calculate a shipping fee from parcel weight and base rate."""
    return weight * base_rate


def apply_coupon_code(total, coupon_percent):
    """Apply a coupon percentage to an order total."""
    return total * (1 - coupon_percent / 100)


def load_audit_record(database, record_id):
    """Load a stored audit record by its identifier."""
    return database.get(record_id)


def save_session_record(database, session_id, record):
    """Save a session record in the database."""
    database[session_id] = record


def get_user_permissions(user_id, permission_store):
    """Read the permission record associated with a user."""
    return permission_store.get(user_id, [])
