;;; Super Metroid 64 x 32 Minimap support.
;;;
;;; Replace old routines and wrap a 32 table into the padding bytes at the top to avoid breaking / hacking everything else.
;;; This also incorporates the seamfix where samus being in the last tile or first tile would inadvertantly wrap and draw wrong tiles.

!bank_90_free_space_start = $90FD40
!bank_90_free_space_end = $90FFFF

org $82B86C
    AND #$00F8          ; change pause map to wrap to row 0
    
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
assert pc() <= !bank_90_free_space_end