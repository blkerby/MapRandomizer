lorom

!size_bytes = #$0E00
!size_words = #$0700

; During door transition, perform VRAM transfers during vblank using NMI,
; instead of during forced blanking (using IRQ).

org $82E039
    LDA $01,s
    TAY
    PHB
    PHK
    PLB

    LDA $0001,Y : STA $00  ; source address
    LDA $0003,Y : STA $02  ; source bank
    LDA $0004,Y : STA $04  ; VRAM destination
    LDA $0006,Y : STA $06  ; size

    PLB

    ; update return address:
    TYA
    CLC
    ADC #$0007
    STA $01,s

    JMP finish_vram_transfer
    
warnpc $82E071

org $82F810
; entry point that assumes the transfer info is already loaded into $00, $02, $04, $06:
finish_vram_transfer:
.loop:
    LDA $800330
    BEQ +
    PHY : JSL $808338 : PLY    ; wait for NMI (since there are pending writes)
+
    LDA $00 : STA $8000D2 : CLC : ADC !size_bytes : STA $00   ; source address
    LDA $02 : STA $8000D4
    LDA $04 : STA $8000D5 : CLC : ADC !size_words : STA $04  ; VRAM destination
    LDA $06       ; size
    CMP !size_bytes
    BCC +
    LDA !size_bytes    ; clamp size to a maximum of $0800 bytes
+
    STA $8000D0

    LDA #$0007 : STA $800330   ; update VRAM write table pointer
 
    ; Update size
    LDA $06
    SEC
    SBC !size_bytes
    STA $06
    BEQ .done
    BMI .done

    BRA .loop

.done:
    RTS

warnpc $82F870


org $82E4ED
    LDA #$008A
    STA $02
    LDA $1964
    BEQ end_fx_load
    STA $00
    LDA #$5BE0
    STA $04
    LDA #$0840
    STA $06
    JSR finish_vram_transfer
    bra end_fx_load
warnpc $82E512
org $82E512
end_fx_load: