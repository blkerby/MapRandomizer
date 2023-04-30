; Patch from Benox (via dagit and moehr)

org $82E659
    JSR handle_door_transition
    NOP

; Free space in bank $82:
org $82FA80
handle_door_transition:
    ; Lets fix the entering from water to no water animation bug 
    STZ $0A9C
    JSL $878064 ; run hi-jacked instruction
    RTS

warnpc $82FA90