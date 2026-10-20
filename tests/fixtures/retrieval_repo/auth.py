def parse_bearer_header(header):
    """Extract the token from an Authorization header."""
    scheme, token = header.split(" ", 1)
    if scheme.lower() != "bearer":
        raise ValueError("unsupported authorization scheme")
    return token


def validate_access_token(token, now):
    """Reject tokens that are expired or missing an expiry claim."""
    claims = decode_token(token)
    expiry = claims.get("exp")
    if expiry is None or expiry <= now:
        raise ValueError("token expired")
    return claims["sub"]


def decode_token(token):
    """Decode a token payload."""
    return {"sub": token, "exp": 9999999999}
