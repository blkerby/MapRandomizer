;; Animated Maptiles for pause map double_item rooms
;; nn_357

lorom
arch 65816

!bank_df_free_space_start = $dfe212 ; this location must match fn write_double_item_table in map_tiles.rs
!bank_df_free_space_end = $dffe3af

!interval = #$0030           
!current_area = $7e1f5b    
!animate_mask = $7ef5a0
!tile_timer = $7ef5a2      
!tile_state = $7ef5a4

;;; Vanilla hooks

org $82913B     ;;; $9120: Pause menu - index 0: map screen ;;;
    jsl AnimateTile
    nop
    nop                    

org $82941e     ;;; ;;; $93C3: Load pause menu map tilemap and area label ;;;
    jsl BuildAnimateBitMask
    nop 
    nop 
    nop 
    nop
    
    
;;; Custom code

org !bank_df_free_space_start

; the contents of this table is modified by fn write_double_item_table in map_tiles.rs
;
; +0	area index [table is generated in area order of the rooms]
;	+2	VRAM tilemap dest
;	+4	tile_a, the priority tile, revealed (palette 7). $FFFF if both items are the same type, so it never matches and tries to flip a tile to an identical tile
;	+6	tile_b, the other item’s tile, revealed (palette 7)
;	+8	byte offset of tile_b’s item in the collected-items array at $7ED870
;	+10	bit mask for tile_b’s item within that byte
;	+12	tile_a for visited tiles (palette 2). $FFFF if same type
;	+14	tile_b for visited tiles (palette 2)

; Padding: unused entries are $FFFF, $0000, $FFFF, $0000, $0000, $0000, $FFFF, $0000.

DoubleItemTable: 
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Billy Mays
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; West Sand Pit
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Watering Hole
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Green Pirates Shaft
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Green Brinstar Reserve
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Norfair Reserve

; Mask layout: !animate_mask uses bit n for “animate entry n” and bit n+8 for “entry n is visited” (VisitedRowMasks).

RowBitMasks:
  dw $0001, $0002, $0004, $0008, $0010, $0020
  
VisitedRowMasks:
  dw $0101, $0202, $0404, $0808, $1010, $2020
    
BuildAnimateBitMask:

  phb
  php
  rep #$30
  pha
  phx
  phy
  phk
  plb

  lda #$0000
  sta !tile_timer
  sta !tile_state
  sta !animate_mask

  ldy #$0000

.loop
  lda.w DoubleItemTable,y
  cmp !current_area
  bne .next_entry

  lda.w DoubleItemTable+8,y
  tax
  lda $7ed870,x
  and.w DoubleItemTable+10,y
  bne .next_entry

  lda.w DoubleItemTable+2,y
  sec
  sbc #$3000
  asl
  tax
  lda $7e4000,x
  cmp.w DoubleItemTable+4,y
  beq .revealed
  cmp.w DoubleItemTable+12,y
  bne .next_entry

  tya
  lsr
  lsr
  lsr
  tax
  lda.w VisitedRowMasks,x
  bra .set_mask

.revealed
  tya
  lsr
  lsr
  lsr
  tax
  lda.w RowBitMasks,x

.set_mask
  ora !animate_mask
  sta !animate_mask

.next_entry
  tya
  clc
  adc #$0010
  tay
  cpy #$0060
  bne .loop

  ply
  plx
  pla
  plp
  plb
  cmp #$0007  ;
  bmi .skip   ; hijacked code. (dont' think this does anything in MapRando anyway, it's for Debug options but restore it regardless.)
  lda #$0000  ;
.skip
  rtl
  

AnimateTile:
  php
  rep #$30
  lda !tile_timer
  inc
  cmp !interval
  bcs .run
  
  sta !tile_timer
  plp
  lda #$0000
  sta $0763
  rtl
  
.run  
  phx
  phy
  phb
  phk
  plb

  
  lda #$0000
  sta !tile_timer
  lda !tile_state
  eor #$0002              ; byte offset of tile_a / tile_b
  sta !tile_state

  lda !animate_mask
  beq .done
  pha                     ; running copy of the mask, shifted once per entry

  ldy #$0000              ; table offset
  lda.l $7e0330
  tax

.loop
  lda $01,s
  lsr
  sta $01,s
  bcc .next               ; bit not set

  lda DoubleItemTable,y
  cmp !current_area
  bne .next

  lda #$0002
  sta $7e00d0,x           ; size
  lda $01,s
  and #$0080
  lsr
  lsr
  lsr
  lsr
  clc
  adc !tile_state
  sta $7e00d2,x
  tya
  clc
  adc #DoubleItemTable+4
  adc $7e00d2,x
  sta $7e00d2,x           ; source = tile_a or tile_b, revealed or visited
  lda #$00df
  sta $7e00d4,x           ; bank (dest write below overwrites the stray high byte)
  lda DoubleItemTable+2,y
  sta $7e00d5,x           ; VRAM dest
  txa
  clc
  adc #$0007
  tax

.next
  tya
  clc
  adc #$0010
  tay
  cpy #$0060
  bne .loop

  txa
  sta.l $7e0330
  pla

.done
  plb
  ply
  plx
  plp
  lda #$0000 
  sta $0763   
  rtl
  
assert pc() <= !bank_df_free_space_end