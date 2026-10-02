; Temporary Sprite Tube test: spawn once after every room's enemy initialization.
; The upstream Sprite Tube.ips supplies the AI and spritemap in bank B2.
arch snes.cpu
lorom

!tube_id = $F7D3
!tube_palette = $0E00 ; Sprite palette 7
!tube_tiles = $01D0   ; Upstream tile base (spritemap uses tile $100)

; Both empty and populated rooms return through this epilogue.
org $A08BE6
    JMP SpawnTube
assert pc() == $A08BE9

; Header fields from Mosaic/Projects/Base/Export/Enemies/F7D3.xml.
; The palette pointer is relative to the AI bank; tables are in bank B4.
org $A0F7D3
TubeHeader:
    dw $8400, TubePalette ; Manual graphics allocation, 1 KiB; palette
    dw $7FFF, $0000       ; Health, contact damage
    dw $0020, $0080       ; X/Y radii
    db $B2, $00           ; AI bank, hurt AI time
    dw $0000, $0000       ; Hurt sound, boss ID
    dw $FEAA, $0001       ; Init AI, number of parts
    dw $0000, $FEF2       ; Unused, main AI
    dw $804C, $804C, $8041 ; Grapple, hurt, frozen AI
    dw $0000, $0000       ; X-ray AI, death animation
    dw $0000, $0000       ; Unused
    dw $804C, $0000       ; Power bomb AI, base instruction list
    dw $0000, $0000       ; Unused
    dw $804C, $804C       ; Touch, shot AI
    dw $0000             ; Unused
    dl TubeGraphics
    db $05               ; Sprite layer
    dw TubeDrops, TubeVulnerabilities, $0000 ; No debug name
assert pc() == $A0F813

org $A0FA00
SpawnTube:
    ; Entry has DB=A0 and 16-bit A/X/Y. Preserve the original return state.
    PHA
    PHX
    PHY
    LDA $0E54
    PHA
    ; Record enemy spawn data ($A0:88D0) clobbers these scratch words.
    LDA $12 : PHA
    LDA $14 : PHA
    LDA $16 : PHA
    LDA $18 : PHA
    LDA $1A : PHA
    LDA $1C : PHA

    LDX #$0000
    LDY #$FFFF ; First free slot, or FFFF if none
.scan:
    LDA $0F78,X
    BEQ .free
    CMP #!tube_id
    BNE .next
    JMP .return ; Do not create a second tube
.free:
    CPY #$FFFF
    BNE .next
    TXY
.next:
    TXA
    CLC
    ADC #$0040
    TAX
    CPX #$0800
    BNE .scan
    CPY #$FFFF
    BNE .spawn
    JMP .return ; All 32 slots are occupied; leave palettes/VRAM alone

.spawn:
    TYX
    STX $0E54
    ; Clear only this slot, including its graphics/spawn-data auxiliary block.
    LDY #$0020
    LDA #$0000
.clear:
    STA $0F78,X
    STA $7E7000,X
    INX
    INX
    DEY
    BNE .clear
    LDX $0E54

    LDA.l TubePopulation+$00 : STA $0F78,X
    LDA.l TubePopulation+$02 : STA $0F7A,X
    LDA.l TubePopulation+$04 : STA $0F7E,X
    LDA.l TubePopulation+$06 : STA $0F92,X
    LDA.l TubePopulation+$08 : STA $0F86,X
    LDA.l TubePopulation+$0A : STA $0F88,X
    LDA.l TubePopulation+$0C : STA $0FB4,X
    LDA.l TubePopulation+$0E : STA $0FB6,X
    LDA.l TubeHeader+$04 : STA $0F8C,X
    LDA.l TubeHeader+$08 : STA $0F82,X
    LDA.l TubeHeader+$0A : STA $0F84,X
    LDA.l TubeHeader+$0C : STA $0FA6,X
    LDA.l TubeHeader+$39
    AND #$00FF
    STA $0F9A,X
    LDA #$0001 : STA $0F94,X
    LDA #!tube_palette
    STA $0F96,X ; Must be set before Tube_Init copies background colors
    STA $7E7008,X
    LDA #!tube_tiles
    STA $0F98,X
    STA $7E7006,X

    TXY
    JSL $A088D0 ; Record initial population fields before AI changes them

    ; Load all 16 supplied colors into the target palette, including black
    ; color 15. Tube_Init then replaces colors 4-7 with background colors.
    LDX #$001E
.palette:
    LDA.l TubePalette,X
    STA $7EC3E0,X
    DEX
    DEX
    BPL .palette

    PHB
    PEA.w $B2B2
    PLB
    PLB
    JSL $B2FEAA ; Retain the spritemap/instruction state established by init
    PLB

    LDX $0E54
    TXA
    CLC
    ADC #$0040
    CMP $0E4C
    BCC .graphics
    STA $0E4C ; Ensure enemy processing also runs in an otherwise empty room
    ; Do not alter population count, kill count, or room-clear threshold.

.graphics:
    ; Queue after normal enemy tile loading. VRAM addresses are word offsets.
    LDX $0330
    LDA #$0400 : STA $00D0,X
    LDA.w #TubeGraphics : STA $00D2,X
    LDA #$00B4 : STA $00D4,X
    LDA #$6D00 : STA $00D5,X
    TXA
    CLC
    ADC #$0007
    STA $0330

.return:
    PLA : STA $1C
    PLA : STA $1A
    PLA : STA $18
    PLA : STA $16
    PLA : STA $14
    PLA : STA $12
    PLA : STA $0E54
    PLY
    PLX
    PLA
    ; Original $A0:8BE6 epilogue (its DB/P were pushed by $A0:8A9E).
    PLB
    PLP
    RTL

TubePopulation:
    ; ID, X, Y, init parameter, properties, extra properties, parameters 1/2
    dw !tube_id, $0080, $0000, $0000, $2800, $0000, $0000, $0000
assert pc() <= $A0FE00

org $B2FFC0
TubePalette:
    incbin "../../Mosaic/Projects/Base/Export/Enemies/F7D3.snes"
assert pc() == $B2FFE0

org $B4F500
TubeDrops:
    db $00, $00, $00, $FF, $00, $00
TubeVulnerabilities:
    ; All 22 entries in the supplied header are 02.
    fillbyte $02
    fill 22
assert pc() <= $B4F600

org $B4F600
TubeGraphics:
    incbin "../../Mosaic/Projects/Base/Export/Enemies/F7D3.gfx"
assert pc() == $B4FA00
