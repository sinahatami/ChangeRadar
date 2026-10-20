def load_user_record(database, user_id):
    """Look up one user record by its identifier."""
    return database.get(user_id)


def save_user_record(database, user_id, record):
    """Persist a user record under its identifier."""
    database[user_id] = record
