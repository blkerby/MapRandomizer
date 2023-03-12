lorom
arch 65816

; Update PLM sets in deactivated states to point to the same as in activated states, to make the items spawn which normally
; do not spawn until Zebes is awake or Phantoon is dead.

; Morph Ball Room (0x9E9F)
org $8F9EC5
    dw $86E6

; Note: Pit Room is handled specially in patch.rs: in this case we can't repoint the
; whole PLM set because it would mess up the gray doors.

; The Final Missile
org $8F9AB6
    dw $8486

; Wrecked Ship West Super Room
org $8FCDCE
    dw $C357

; Wrecked Ship East Super Room
org $8FCE17
    dw $C35F

; Wrecked Ship Energy Tank Room
org $8FCC4D
    dw $C337

; Assembly Line
org $8FCAD4
    dw $C319

; Bowling Alley
org $8FC9B4
    dw $C2D1

; Gravity Suit Room
org $8FCE66
    dw $C36D