;;; Super Metroid 64 x 32 Minimap support.
;;;
;;; Replace old routines and wrap a 32 table into the padding bytes at the top to avoid breaking / hacking everything else.
;;; This also incorporates the seamfix where samus being in the last tile or first tile would inadvertantly wrap and draw wrong tiles.
;;; New custom clamp limits and adjusted offsets stop the pausemap wrapping around, the maridia clamp should be unaffected.

!bank_90_free_space_start = $90FD40
!bank_90_free_space_end = $90FFFF

;;;
;;; Vanilla hooks
;;;

;;; Bank 82

org $82B86C
    AND #$00F8                ; change pause map to wrap to row 0
org $8290B3
    db $0D                      ; BMI now targets $90C1 (was $90C7)
org $8290BD
    JML UpClampThenDown         ; was: CMP #$FFD8 : BPL : LDA #$FFD8 : STA $B3
org $8290C1
    JML DownOnly                ; reached from the BMI above
org $82A009
    JML TopRow                ; replaces "Determine topmost map row"
org $82A053
    JML BottomRow             ; replaces "Determine bottommost map row"
org $829F3F                   ; was: STA $05B2 : PLB : PLP : RTS
    JML ClampLimits
    NOP : NOP
org $829E74                   ; was: LDA $05B2 : SEC
    JSL ClampB1

;;; Bank 90
    
org $90A8D4
    JSR WrapRow         ; was: INC A : STA $16    
org $90AB5F
    JMP MarkAbove       ; rest of old routine ($AB62-$AB74) is now dead
org $90A970             ; wrap the Samus row so real row 31 -> tilemap row 0
    JSR WrapRow         ; was: INC A : STA $16
org $90A9A2             ; skip the old explored-bit reads, keep the area-pointer setup at $A9BD
    db $80,$19
org $90AA98
    JSR FixBottom
org $90A9D6             ; replace the old ROM-map reads and spill call
    JSR FetchWindow
    JMP $AA2C           ; BRANCH_SINGLE_PAGE
WrapRow:
    INC A
    AND #$001F
    STA $16
    RTS
MarkAbove:
    PHX : PHY
    LDA $1E
    AND #$007C         ; tilemap row * 4 within the page
    CMP #$0004         ; tilemap row 1 = real row 0
    BEQ .done          ; nothing above real row 0, so skip
    SEC : SBC #$0004   ; row - 1
    AND #$007C         ; wraps tilemap row 0 -> 31 (real row 31 -> 30)
    PHA
    LDA $1E
    AND #$0083         ; keep page bit and column byte
    ORA $01,s          ; recombine with the new row
    TAX                ; X = in-area byte index of the tile above (0..$FF)
    PLA
    SEP #$20
    LDY $20
    LDA $07F7,x
    ORA $AC04,y
    STA $07F7,x

    ; mirror to SRAM: in-area index ($00-$FF) -> overall index (area*$100 + index)
    TXA : XBA
    LDA $1F5B
    XBA : TAX          ; 16-bit TAX copies the full accumulator: X = area<<8 | index
    LDA $702000,x
    ORA $AC04,y
    STA $702000,x

    REP #$20
.done:
    PLY : PLX
    RTS
assert pc() <= $90AA2C

org $90AB75            ; ;;; $AB75: Adjust map bits for map page spill ;;; Rewrite this whole terrible thing.
FetchWindow:
    LDA $0F : STA $09           ; $0B is still $82 from $A9C6
    LDA $22 : LSR : LSR : LSR   ; page * 4
    CLC : ADC $14
    SEC : SBC $2E
    STA $32                     ; global byte column of origin (-1..7)
    LDA $16 : DEC : AND #$001F : DEC
    STA $30                     ; real row above Samus (-1 = off map)
    LDX #$0000
.row:
    LDA $30 : CMP #$0020 : BCS .blank
    LDA $32 : JSR .get
    XBA : STA $26,x
    LDA $2C : XBA : STA $18,x
    LDA $32 : INC
    JSR .get
    ORA $26,x : STA $26,x
    LDA $2C : ORA $18,x
    LDY $34 : AND $AC0C,y
    STA $18,x
.next:
    INC $30 : INX : INX
    CPX #$0006 : BCC .row
    RTS
.blank:
    STZ $18,x : STZ $26,x
    BRA .next

.get:                           ; A = global byte column; returns map byte in A, explored byte in $2C
    STZ $2C
    CMP #$0008 : BCS .off
    PHA
    AND #$0003 : STA $2C
    LDA $30 : INC : AND #$001F : ASL : ASL
    ORA $2C : STA $2C
    PLA
    AND #$0004
    ASL : ASL : ASL : ASL : ASL
    ORA $2C : TAY
    LDA $07F7,y : AND #$00FF : STA $2C
    LDA [$09],y : AND #$00FF
    RTS
.off:
    LDA #$0000
    RTS

assert pc() <= $90AC04

;;;
;;; Custom code in free space
;;;

org !bank_90_free_space_start
FixBottom:
    LDA $16 : BEQ .bottom        ; real row 31: window rows 31,0,1
    CMP #$001F : BNE .done       ; real row 30: window rows 30,31,0
    LDA $06 : SEC : SBC #$0800 : STA $06
    BRA .done
.bottom:
    TYA : CLC : ADC #$0800 : TAY
    LDA $03 : SEC : SBC #$0800 : STA $03
    LDA $06 : SEC : SBC #$0800 : STA $06
.done:
    LDA #$0005
    RTS

TopRow:                       ; $00 = map base. Skips tilemap row 0, scans rows 1-30
    PHP
    REP #$20
    LDA $00 : CLC : ADC #$0004 : STA $00
    CLC : ADC #$0080 : STA $03
    SEP #$20
    LDA $02 : STA $05
    LDX #$0001
    LDY #$0000
.loop:
    LDA [$00],y : BNE .done
    LDA [$03],y : BNE .done
    INY : CPY #$0004 : BMI .loop
    LDY #$0000
    REP #$20
    LDA $00 : CLC : ADC #$0004 : STA $00
    LDA $03 : CLC : ADC #$0004 : STA $03
    SEP #$20
    INX : CPX #$001F : BMI .loop
    LDX #$0001
.done:
    JML $82A051               ; PLP : RTS

BottomRow:                    ; $06 = map base. Row 0 counts as row 32, then rows 31..1
    PHP
    REP #$20
    LDA $06 : STA $00
    CLC : ADC #$0080 : STA $03
    SEP #$20
    LDA $02 : STA $05
    LDY #$0000
.row0:
    LDA [$00],y : BNE .row32
    LDA [$03],y : BNE .row32
    INY : CPY #$0004 : BMI .row0
    REP #$20
    LDA $06 : CLC : ADC #$007C : STA $00
    CLC : ADC #$0080 : STA $03
    SEP #$20
    LDX #$001F
    LDY #$0000
.loop:
    LDA [$00],y : BNE .done
    LDA [$03],y : BNE .done
    INY : CPY #$0004 : BMI .loop
    LDY #$0000
    REP #$20
    LDA $00 : SEC : SBC #$0004 : STA $00
    LDA $03 : SEC : SBC #$0004 : STA $03
    SEP #$20
    DEX : BNE .loop
    LDX #$000B
.done:
    JML $82A098               ; PLP : RTS
.row32:
    LDX #$001F
    BRA .done

ClampLimits:                  ; A = max Y scroll, as in vanilla
    STA $05B2
    LDA $05AC : CMP #$0010 : BPL +
    LDA #$0010 : STA $05AC    ; min X >= 10h, so X scroll >= 0
+   LDA $05AE : CMP #$01E9 : BMI +
    LDA #$01E8 : STA $05AE    ; max X <= 1E8h, so X scroll <= 100h
+   LDA $05B0 : CMP #$0010 : BPL +
    LDA #$0010 : STA $05B0    ; min Y >= 10h, so Y scroll >= -20h
+   PLB : PLP
    JML $829F44               ; RTS

ClampB1:
    LDA $B1 : BPL .pos
    CMP #$FFF8 : BPL .done
    LDA #$FFF8 : STA $B1 : BRA .done
.pos:
    CMP #$0101 : BCC .done
    LDA #$0100 : STA $B1
.done:
    LDA $05B2 : SEC           ; replaced instructions
    RTL
UpClampThenDown:                          ; A = [BG1 Y scroll] here
    CMP #$FFD8 : BPL DownOnly
    LDA #$FFD8 : STA $B3                  ; BG1 Y scroll = max(scroll, -28h)
DownOnly:
    LDA $0AFA : XBA : AND #$00FF
    CLC : ADC $07A3 : INC
    ASL : ASL : ASL                       ; Samus pixel Y, same as $90A4-$90A6
    SEC : SBC $B3                         ; screen Y
    CMP #$00B8 : BMI .done                ; visible enough, leave it alone
    SEC : SBC #$00B8
    CLC : ADC $B3 : STA $B3               ; scroll down by the excess
    LDA $05B2 : SEC : SBC #$00B1          ; don't go past what the down arrow allows
    AND #$FFF8 : CLC : ADC #$0008
    CMP $B3 : BPL .done
    STA $B3
.done:
    RTL                                   ; $9028 is entered with JSL

print pc
assert pc() <= !bank_90_free_space_end