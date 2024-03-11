lorom

!bank_84_free_space_start = $84F700
!bank_84_free_space_end = $84F730

org !bank_84_free_space_start

; This PLM entries must be at an address matching what is in `patch.rs`, starting here at $84F700
dw $EE64, inst        ; PLM $F700 (nothing, for Bomb Torizo Room)

inst:
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
    dw $E04F                               ; Draw item frame 0 (not sure why we need to do this, but it doesn't work otherwise)
    dw $86B4                               ; sleep
.triggered:
    dw $8899                               ; Set the room argument item
    dw $8724, $DFA9                        ; Go to $DFA9

warnpc !bank_84_free_space_end