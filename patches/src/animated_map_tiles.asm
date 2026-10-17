;; Animated Maptiles for pause map double_item rooms
;; nn_357

lorom
arch 65816

!bank_df_free_space_start = $dfe212 ; values in this table are overwritten in patch.rs
!bank_df_free_space_end = $dffeef

!tile_timer = $7e09ee      
!tile_state = $7e09a0
!interval   = #$0030           
!current_area = $7e1f5b    
!animate_mask = $7ef5a0    

; Vanilla hooks

org $82913B     ;;; $9120: Pause menu - index 0: map screen ;;;
    jsl AnimateTile
    nop
    nop                    

org $82941e     ;;; ;;; $93C3: Load pause menu map tilemap and area label ;;;
    jsl BuildAnimateBitMask
    nop ;$82:941E C9 07 00    CMP #$0007             ;\
    nop ;$82:9421 30 03       BMI $03    [$9426]     ;} If [area index] >= debug:
    nop ;$82:9423 A9 00 00    LDA #$0000             ; A = 0
    nop
    
    
; Custom code

org !bank_df_free_space_start

; area, vram_dest, tile_a, tile b
; tile_a will be set to FFFF on the server side if both items are of the same type so it is never animated as it will just waste cycles reloading an identical tile.
; table is ordered by area for the 6 double_item rooms, upto 6, if small maps / missing etc table is padded with FFFF to fill this table.

DoubleItemTable:
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Billy Mays
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; West Sand Pit
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Watering Hole
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Green Pirates Shaft
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Green Brinstar Reserve
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Norfair Reserve


RowBitMasks:
  dw $0001, $0002, $0004, $0008, $0010, $0020
    
BuildAnimateBitMask:

  ; entry
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

  lda.w DoubleItemTable+2,y
  sec
  sbc #$3000
  asl
  tax

  lda $7e4000,x
  and #$E3FF
  cmp.w DoubleItemTable+4,y
  bne .next_entry

  lda.w DoubleItemTable+8,y
  tax
  lda $7ed870,x
  and.w DoubleItemTable+10,y
  bne .next_entry

  tya
  lsr
  lsr
  lsr
  tax
  
  lda.w RowBitMasks,x
  ora !animate_mask
  sta !animate_mask

.next_entry
  tya
  clc
  adc #$0010
  tay
  cpy #$0060
  bne .loop
  
  ; cleanup
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

  phx
  lda DoubleItemTable+2,y
  sec
  sbc #$3000
  asl
  tax
  lda $7e4000,x
  and #$1C00
  pha
  lda !tile_state
  bne .tile_b
  lda DoubleItemTable+4,y
  bra .merge
.tile_b
  lda DoubleItemTable+6,y
.merge
  ora $01,s
  sta $7e4000,x
  pla
  txa
  clc
  adc #$4000
  plx
  sta $7e00d2,x
  lda #$0002
  sta $7e00d0,x
  lda #$007E
  sta $7e00d4,x           ; bank (dest write below overwrites the stray high byte)
  lda DoubleItemTable+2,y
  sta $7e00d5,x
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
  
  print pc
  assert pc() <= !bank_df_free_space_end