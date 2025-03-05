; Modify the routines that load level data, removing the parts that initialize
; the level data and copy the BTS and Layer2 to their destination. We are using
; a format for the compressed level data which already includes the initialization
; and the correct location for BTS and Layer2.

namespace DOOR_TRANSITION
org $82EA78
    BRA skip_init
org $82EA92
skip_init:
org $82EAA3
    BRA skip_copy
org $82EADB
skip_copy:
namespace off

namespace NEW_GAME
org $82E7DC
    BRA skip_init
org $82E7EA
skip_init:
org $82E7FB
    BRA skip_copy
org $82E833
skip_copy:
namespace off