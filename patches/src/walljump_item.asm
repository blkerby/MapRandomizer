lorom

!bank_84_free_space_start = $84F600
!bank_84_free_space_end = $84F700

org !bank_84_free_space_start

; These PLM entries must be at an address matching what is in `patch.rs`, starting here at $84F600
dw $EE64, inst    ; PLM $F600 (wall-jump boots)

;;; Instruction list - PLM $F600 (wall-jump boots)
inst:
    dw $8764, $9100                        ; Load item PLM GFX
    db $01, $01, $01, $01, $01, $01, $01, $01
    dw $887C, .end                         ; Go to end if the room argument item is set
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
.animate:
    dw $E04F                               ; Draw item frame 0
    dw $E067                               ; Draw item frame 1
    dw $8724, .animate                     ; Go to $E1FD
.triggered:
    dw $8899                               ; Set the room argument item
    dw $8BDD                               ; Clear music queue and queue item fanfare music track
    db $02                            
    dw $88F3, $2000                        ; Pick up equipment 2000h and display message box Dh
    db $0D
.end
    dw $8724, $DFA9                        ; Go to $DFA9

warnpc !bank_84_free_space_end