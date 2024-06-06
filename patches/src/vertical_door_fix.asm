!bank_84_free_space_start = $84F4A0
!bank_84_free_space_end = $84F500


; red door facing down - PLM instruction list:
org $84C43E
red_door_down_inst:
    dw $8A72, $C544               ; Go to $C544 if the room argument door is set
    dw $8A24, red_door_down_shot  ; Link instruction = red_door_down_shot
    dw $86C1, $BD50               ; Pre-instruction = go to link instruction if shot with a (super) missile
    ; Run initial draw instruction twice, otherwise it would fail to get drawn in Red Brinstar Elevator Room and Forgotten Highway Elevator:
    dw $0002, $A977
    dw $0001, $A977
red_door_down_sleep:
    dw $86B4           ; Sleep
red_door_open:
    ; Queue sound 7, sound library 3, max queued sounds allowed = 6 (door opened)
    dw $8C19
    db $07
    dw $0006, $A983
    dw $0006, $A98F
    dw $0006, $A99B
    dw $0001, $A69B
    dw $86BC            ; Delete
warnpc $84C489

; green door facing down - PLM instruction list:
org $84C2B9
green_door_down_inst:
    dw $8A72, $C544      ; Go to $C544 if the room argument door is set
    dw $8A24, green_door_down_shot   ; Link instruction = green_door_down_shot
    dw $86C1, $BD88      ; Pre-instruction = go to link instruction if shot with a super missile
    ; Run initial draw instruction twice, otherwise it would fail to get drawn in Red Brinstar Elevator Room and Forgotten Highway Elevator
    dw $0002, $A8B7
    dw $0001, $A8B7
    dw $86B4            ; Sleep
green_door_down_shot:
    ; Increment door hit counter; Set room argument door and go to `green_door_open` if [door hit counter] >= 01h
    dw $8A91
    db $01
    dw green_door_open
    ; Skip handling other case (door counter < 1) since this can't happen.
green_door_open:
    ; Queue sound 7, sound library 3, max queued sounds allowed = 6 (door opened)
    dw $8C19
    db $07
    dw $0006, $A8C3
    dw $0006, $A8CF
    dw $0006, $A8DB
    dw $0001, $A69B
    dw $86BC            ; Delete
warnpc $84C301

; If needed, we could probably fit this into unused parts of green/yellow door instructions instead of using free space.
org !bank_84_free_space_start
red_door_down_shot:
    ; Increment door hit counter; Set room argument door and go to `red_door_open` if [door hit counter] >= 05h
    dw $8A91
    db $05
    dw red_door_open   
    ; Queue sound 9, sound library 3, max queued sounds allowed = 6 (missile door shot with missile)
    dw $8C19
    db $09
    dw $0003, $AA67
    dw $0004, $A977
    dw $0003, $AA67
    dw $0004, $A977
    dw $0003, $AA67
    dw $0004, $A977
    dw $8724, red_door_down_sleep      ; Go to red_door_down_sleep
warnpc !bank_84_free_space_end
