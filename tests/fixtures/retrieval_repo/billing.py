def calculate_tax(subtotal, tax_rate):
    """Calculate sales tax from a subtotal and decimal tax rate."""
    return round(subtotal * tax_rate, 2)


def apply_discount(subtotal, discount_percent):
    """Apply a percentage discount without producing a negative amount."""
    discounted = subtotal * (1 - discount_percent / 100)
    return max(discounted, 0)
