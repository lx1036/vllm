import enum

notifs = {"a": "b"}
for notif, value in notifs.items():
    print(notif, value)


class NixlRole(enum.Enum):
    """
    Enum to represent the role of the Nixl connection.
    """
    SENDER = "sender"
    RECEIVER = "receiver"

# NixlRole.SENDER NixlRole.SENDER
print(NixlRole.SENDER, str(NixlRole.SENDER))
