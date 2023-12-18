; Keep escape timer visible when escaping Zebes with gunship.
; This solution keeps the escape digits in the VRAM by preventing the gunship dust clouds to be drawn,
; which makes it easy to keep the time visible on screen.

lorom

org $90E908
    JSR HOOK_ENTER_EXIT_GUNSHIP

org $90F900
HOOK_ENTER_EXIT_GUNSHIP:
    LDA $0943   ;\
    BEQ $04     ;} If [timer status] != inactive:
    JSL $809F6C ; Draw Timer
    JSR $EA7F ; Original code (Low health check)
    RTS

; Don't write gunship smoke clouds to VRAM
org $A2ABC7
    JMP $ABF0

; Don't spawn gunship smoke clouds
org $A2AC95
    RTL
