from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
