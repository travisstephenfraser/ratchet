import math


def mapping(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def finite_number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def rate(value, field):
    number = finite_number(value, field)
    if not 0 <= number <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def whole_number(value, field, *, minimum, maximum=None):
    if isinstance(value, bool):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{field} must be an integer >= {minimum}{upper}")
    number = finite_number(value, field)
    if not number.is_integer() or number < minimum or (maximum is not None and number > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{field} must be an integer >= {minimum}{upper}")
    return int(number)


def nonblank(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value.strip()
