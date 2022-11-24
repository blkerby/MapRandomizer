from logic.rooms import brinstar_red
from logic.rooms import brinstar_blue
from logic.rooms import brinstar_green
from logic.rooms import brinstar_pink
from logic.rooms import brinstar_warehouse
from logic.rooms import crateria
from logic.rooms import maridia_outer
from logic.rooms import maridia_inner
from logic.rooms import norfair_lower
from logic.rooms import norfair_upper
from logic.rooms import wrecked_ship
from logic.rooms import tourian

# TODO: fix in sm-json-data:
# Fix door address for Green Brinstar Main Shaft door to Firefleas: should be 0x18CCA
# Fix door address for Maridia Missile Refill door (to Halfie Climb Room): should be 0x1A894 (not 0x1A8F4)
# Fix name of door to Magdollite Tunnel in Purple Shaft

rooms = (
        crateria.rooms +
        brinstar_red.rooms +
        brinstar_blue.rooms +
        brinstar_green.rooms +
        brinstar_pink.rooms +
        brinstar_warehouse.rooms +
        maridia_outer.rooms +
        maridia_inner.rooms +
        norfair_lower.rooms +
        norfair_upper.rooms +
        wrecked_ship.rooms +
        tourian.rooms
)
