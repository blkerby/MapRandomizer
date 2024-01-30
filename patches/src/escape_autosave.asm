; Auto-save so that the escape can be retried without fighting Mother Brain again.
;
; Author: maddo

lorom

!bank_84_freespace_start = $84FC40
!bank_84_freespace_end = $84FCC0


; Hi-jack code that runs just before the escape starts
;org $A9B216
org $A9B1F7
    JSR autosave

; New save station data (at Tourian save station index 2, which is unused by the vanilla game)
org $80CA4B
    dw $DD58  ; Room ID for Mother Brain Room
    dw door_ptr
    dw $0000  ; Door BTS (unused)
    dw $0000  ; Save X position
    dw $0000  ; Save Y position
    dw $00BF  ; Samus Y position
    dw $0000  ; Samus X position

; Free space in bank $83
org $83AD66
door_ptr:
    dw $DD58
    db $00
    db $04
    db $01
    db $06
    db $00
    db $00
    dw $8000
    dw door_asm

; Free space in bank $84
; PLMs to clear the room (e.g., remove the pipes around where Mother Brain's tank would be)
org !bank_84_freespace_start

clear_row_plm:
    dw $B3D0, clear_row_inst

clear_row_inst:
    dw $0001, clear_row_draw
    dw $0001, clear_row_draw
    dw $86BC

clear_row_draw:
    dw $000A, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $0000

ceiling_plm:
    dw $B3D0, ceiling_inst

ceiling_inst:
    dw $0001, ceiling_draw
    dw $0001, ceiling_draw
    dw $86BC

ceiling_draw:
    dw $0004, $12FC, $1243, $1244, $12FC, $0000

floor_plm:
    dw $B3D0, floor_inst

floor_inst:
    dw $0001, floor_draw
    dw $0001, floor_draw
    dw $86BC

floor_draw:
    dw $0006, $124B, $1339, $124C, $124D, $1339, $124E, $0000

warnpc !bank_84_freespace_end

; Overwrite main ASM for Mother Brain room state with Mother Brain dead (unused state in vanilla game)
org $8FDDB4
    dw main_asm

;; Overwrite enemy GFX and enemy population for Mother Brain room state with Mother Brain room dead
;; (Even though Mother Brain is not visible, we need the enemy to spawn in order to execute the escape sequence.)
org $8FDDAA
    dw enemy_population
    dw $9102


; Free space in bank $A1
org $A1F400
; Custom enemy population which includes Mother Brain body and brain but not Rinkas (nor Zebetites):
; We need these Mother Brain "enemies" to spawn in order for the escape sequence to work, even though
; they will not be visible.
enemy_population:
    dw $EC7F, $0081, $006F, $0000, $2800, $0004, $0000, $0000
    dw $EC3F, $0081, $006F, $0000, $2800, $0004, $0000, $0000
    dw $FFFF
    db $00
warnpc $A1F440

org $A986EE
    JSR body_init
    RTL

org $A9873A
    JSR brain_init

; When loading escape auto-save (playing elevator room music), use escape music song set to reduce lag when switching tracks:
org $8FDDA6
    db $FF, $03

; Free space in bank $8F
org $8FF800
main_asm:
    LDA $0A1E
    BEQ .skip   ; Samus is still facing forward (initial state after loading), so do not trigger escape yet

    LDA #$000E
    JSL $808233
    BCS .skip   ; Escape already triggered, so don't trigger again

    ; Trigger escape:
    LDA #$000E
    JSL $8081FA ; Set escape flag

    LDA #$0000             ;\
    JSL $808FC1            ;} Queue music stop
    LDA #$FF24             ;\
    JSL $808FC1            ;/ Queue escape music

    LDA #$B211             ; Mother Brain's body function = $B211 (32 frame delay))
    STA $0FA8 
    LDA #$0020             ;\
    STA $0FB2              ;} Mother Brain's body function timer = 20h

.skip
    RTS

door_asm:
    LDA $7ECD20
    AND #$00FF             ;} Scroll 1 = red
    STA $7ECD20

    ; Seal right wall
    JSL $8483D7
    db $0F, $04
    dw $B673
    
    JSL $8483D7
    db $0F, $09
    dw $B673

    ; Clear the middle of the room
    JSL $8483D7
    db $06, $02
    dw ceiling_plm

    JSL $8483D7
    db $05, $03
    dw clear_row_plm

    JSL $8483D7
    db $05, $04
    dw clear_row_plm

    JSL $8483D7
    db $05, $09
    dw clear_row_plm

    JSL $8483D7
    db $05, $0A
    dw clear_row_plm

    JSL $8483D7
    db $05, $0B
    dw clear_row_plm

    JSL $8483D7
    db $05, $0C
    dw clear_row_plm

    JSL $8483D7
    db $05, $0D
    dw floor_plm

    RTS

; Free space in bank $A9
ORG $A9FD40
autosave:
    LDA #$0001
    JSL $8081A6 ; set main boss bit for current area (mark Mother Brain as dead)

    LDA #$0002
    STA $078B   ; Set save station index to 2
    LDA $0952   ; Use current save slot (which will be incremented by rolling-save code in saveload.asm)
    JSL $818000

    ;STZ $0FBA   ; run hi-jacked instruction
    LDA #$0000
    RTS

body_init:
    LDA #$0001
    JSL $8081DC   ; Check if Mother Brain is marked dead (indicating we're respawning in escape). 
    BCS .skip     ; If so, skip loading FX entry (acid) and spawning turrets.
    
    ; Finish vanilla Mother Brain body initialization:
    LDA #$0001
    JSL $89AB02            ;} Load FX entry index 1
    LDA #$0000             ;\
                           ;|
    PHA                    ;|
    LDY #$C17E             ;|
    JSL $868097            ;} Spawn Ch Mother Brain's room turret enemy projectiles
    PLA                    ;|
    INC A                  ;|
    CMP #$000C             ;|
    BCC $F1                ;/
.skip:

    RTS

brain_init:
    LDA #$0001
    JSL $8081DC   ; Check if Mother Brain is marked dead (indicating we're respawning in escape). 
    BCC .skip     ; If not, continue with normal brain initialization.
    
    ; During respawn in escape room, move brain off-screen:
    STZ $0FBA   ; Mother Brain's brain X position = 0
    STZ $0FBE   ; Mother Brain's brain Y position = 0

    ; Acquire hyper beam
    LDA #$0003
    JSL $91E4AD    
.skip
    JSR $D1F8   ; run hi-jacked instruction
    RTS

warnpc $A9FE00