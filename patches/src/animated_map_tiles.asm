;; Animated Maptiles for pause map double_item rooms
;; nn_357

lorom
arch 65816

incsrc "constants.asm"

!bank_df_free_space_start = $dfe212 ; this location must match fn write_double_item_table in map_tiles.rs
!bank_df_free_space_end = $dfe3c2
!bank_90_free_space_start = $90fd40
!bank_90_free_space_end = $90ff40

!interval = #$001B
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
  dw $0000, $0000, $0000, $0000, $0000, $0000, $0000, $0000 ; Bomb Torizo (when it's an objective)
DoubleItemTableEnd:

; Mask layout: !animate_mask uses bit n for “animate entry n” and bit n+8 for “entry n is visited” (VisitedRowMasks).

RowBitMasks:
  dw $0001, $0002, $0004, $0008, $0010, $0020, $0040

VisitedRowMasks:
  dw $0101, $0202, $0404, $0808, $1010, $2020, $4040
    
BuildAnimateBitMask:

  phb
  php
  rep #$30
  pha
  phx
  phy
  phk
  plb

  lda #$0015
  sta !tile_timer
  lda #$0000
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
  cpy #DoubleItemTableEnd-DoubleItemTable
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
  cpy #DoubleItemTableEnd-DoubleItemTable
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

;;; Animated minimap patch: alternate item icons when they differ
;;;
;;; Stag Shot  

org $90aa43
    jsr check_coords

org !bank_90_free_space_start
; pre_minimap_hook
; table @ $dfb000 (written by map_tiles.rs)
;   db area, dw x, dw y, db item_1_byte, dw item_1_bit, db item_2_byte, dw item_2_bit, dw offset_room_ptrs
;       room_ptrs table: dw room_hdr, dw tile_a, dw tile_b .... 0x0000 terminated
; $7ef5b0 = animated tile flag: ffff = new room, 0 = no match, 1 = match, 2 = only update HUD offsets)
; $7ef5b2 = curr map x
; $7ef5b4 = curr map y
; $7ef5b6 = room relevance bitfield (either: room has double-item or is adjacent to one)
; $7ef5c0 = current animation table
;   format: db hud_tile_offset, dw reveal_offset, dw reveal_bit, dw tile_a, dw tile_b .... 0xffff terminated
!animated_tiles = $dffb00

check_coords:
; save x/y for post_minimap_hook
    lda $af6
    and #$ff00
    xba
    clc
    adc $7a1
    sta !minimap_settings+2 ; true x coord
    lda $16
    sta !minimap_settings+4 ; y coord
    
    lda !minimap_settings
    cmp #$ffff              ; new room?
    bne .exit
    ldx #$0000              ; table ptr
    txy                     ; i

.lp
    phx                     ; table ptr
    tya
    asl
    tax
    lda $8085b6,x           ; dw 0001, 0002, 0004, 0008, 0010, 0020, 0040, 0080
    and !minimap_settings+6 ; check relevance bitmask
    bne .match
    plx                     ; table ptr
    bra .next
.match
    plx                     ; table ptr
    lda #$0001              ; set match flag
    bra .done
    
.next
    iny                     ; i++
    cpy #$0006
    beq .no_match
    txa
    clc
    adc #$0011              ; next table entry
    tax
    bra .lp

.no_match
    lda #$ffff
    sta !minimap_animate_table ; null terminate start of table for map_progress_maintain
    inc                     ; no match
.done
    sta !minimap_settings
.exit
    lda $16                 ; replaced code
    clc                     ;
    rts

;;; post-minimap hook

hook_post_minimap:
    php                     ; save flags for replaced code
    lda !minimap_settings   ; 0 = no match, 1 = match, 2 = HUD offset updates
    bne .do_anim
.exit_jmp
    jmp .exit
.do_anim
    dec                     ; 1?
    beq .build_table
; only do HUD offset updates
    ldx #$0000
    lda !minimap_settings+2 ; move map x/y to DP (save cycles)
    sta $12
    lda !minimap_settings+4
    sta $16
.hud_lp
    lda !minimap_animate_table,x
    cmp #$ffff              ; end of table?
    beq .exit_jmp
; hud coord is x*2 + y*0x40
    sec
    sbc $12                 ; x-coord
    inc : inc
    asl
    sta !minimap_animate_table+4,x
    lda !minimap_animate_table+2,x
    sec
    sbc $16                 ; y-coord
    inc
    asl
    phx                     ; table ptr
    tax
    lda $cd08,x             ; dw 0000, 0040, 0080
    plx                     ; table ptr
    clc
    adc !minimap_animate_table+4,x
; sanity check location
    pha                     ; hud offset
    cmp #$0089              ; max is 0x88
    bcs .invalid
    and #$003f
    cmp #$000a              ; ensure < 10
    bcs .invalid
    and #$0001              ; and even
    bne .invalid
    pla                     ; hud offset
    bra .write
.invalid
    pla                     ; hud offset
    lda #$ffff
.write
    sta !minimap_animate_table+4,x
    txa
    clc
    adc #$000e              ; table entry++
    tax
    bra .hud_lp
   
; only runs once at end of door transition
.build_table
    ldx #$0000
    txy
    lda !minimap_settings+2 ; move map x/y to DP (save cycles)
    sta $12
    lda !minimap_settings+4
    sta $16
    lda #(!minimap_animate_table&$ffff)
    sta $08                 ; animation table ptr
    lda #((!minimap_animate_table>>16)&$ff)
    sta $0a
    
.build_lp
    phx                     ; table ptr
    tya
    asl
    tax
    lda !minimap_settings+6
    and $8085b6,x           ; dw 0001, 0002, 0004, 0008, 0010, 0020, 0040, 0080
    bne .revealed
    plx                     ; table ptr
    jmp .next
    
.revealed
    plx                     ; table ptr
    lda !animated_tiles,x
    and #$00ff
    cmp $1f5b               ; area match?
    beq .check_item_bit
    jmp .next

; see if item(s) are already acquired
.check_item_bit
    lda !animated_tiles+5,x
    and #$00ff
    phx                     ; table ptr
    tax
    lda $7ED870,x
    plx                     ; table ptr
    and !animated_tiles+6,x
    bne .item_acquired
    lda !animated_tiles+8,x
    and #$00ff
    phx                     ; table ptr
    tax
    lda $7ED870,x
    plx                     ; table ptr
    and !animated_tiles+9,x
    beq .match
.item_acquired
    jmp .next

.match
; now get offset and check which room for appropriate tiles
    phx                     ; table ptr
    phy                     ; i
    lda !animated_tiles+15,x ; rooms ptr
    txy
    tax
.rooms_lp
    lda $df0000,x           ; room from table
    beq .not_found_jmp
    cmp $79b                ; curr room
    beq .found_room
    txa
    clc
    adc #$0006              ; room++
    tax
    bra .rooms_lp
.not_found_jmp
    jmp .not_found
    
.found_room
    phx                     ; table ptr
    tyx
    lda !animated_tiles+1,x
    sta [$08]               ; save x
    inc $08 : inc $08
    pha                     ; x
    lda !animated_tiles+3,x
    sta [$08]               ; save y
    inc $08 : inc $08
    sec
    sbc $16                 ; y-coord
    inc
    asl
    pha                     ; y*2
    cmp #$0005
    bcs .out_of_range       ; reject > 4
    and #$0001
    bne .out_of_range       ; reject odd values
    plx                     ; y*2
    lda $cd08,x             ; existing table with: 0000, 0040, 0080
    sta [$08]
    pla                     ; x
    sec
    sbc $12                 ; x-coord
    inc : inc
    asl
    pha                     ; x*2
    cmp #$0009              ; reject > 8
    bcs .out_of_range2
    and #$0001
    bne .out_of_range2      ; reject odd values
    pla                     ; x*2
    clc
    adc [$08]
    bra .write2
.out_of_range
    pla                     ; fix stack
.out_of_range2
    pla                     ; fix stack
    lda #$ffff
.write2    
    sta [$08]               ; validated HUD offset (x*2 + y*40)
    inc $08 : inc $08
    tyx
    lda $1f5b
    xba
    clc
    adc !animated_tiles+11,x
    sta [$08]               ; bit offset
    inc $08 : inc $08
    lda !animated_tiles+13,x
    sta [$08]               ; bit
    inc $08 : inc $08 
    plx                     ; table ptr
    lda $df0000+2,x
    sta [$08]               ; tile a
    inc $08 : inc $08
    lda $df0000+4,x
    sta [$08]               ; tile b
    inc $08 : inc $08
    
.not_found
    ply                     ; i
    plx                     ; table ptr
    
.next
    iny                     ; i++
    cpy #$0006
    beq .leave
    txa
    clc
    adc #$0011              ; table entry++
    tax
    jmp .build_lp

.leave
    lda #$ffff
    sta [$08]               ; terminate anim table
    lda #$0002
    sta !minimap_settings   ; HUD offset updates
    
.exit
    plp
    bne .no_flash           ; replaced code
    lda $7ec680
    jmp $ab56               ; flash mask
.no_flash
    jmp $ab5d               ; plp / rtl

assert pc() <= !bank_90_free_space_end
    
org $90ab50
    jmp hook_post_minimap : nop : nop : nop

; Use door transition to reset coords check and also update room_relevance bitfield.
; This offloads a lot of computation to minimize minimap update cycles.
; $79b is already set to new room at this point
org $82de49
    jsr hook_door_local : nop

org $82be5a                 ;;; unused vanilla code; if this is moved, update item_dots_disappear.asm
    jsr hook_door
    rtl
    
hook_door_local:
    jsr hook_door
    stz $12                 ; replaced code
    stz $14                 ;
    rts
    
hook_door:
    php
    phx
    phy
    
    lda #$ffff              ; new room
    sta !minimap_settings
    inc
    sta !minimap_settings+6 ; clear relevance bitfield
    tay                     ; bit ptr
    tax                     ; table ptr
; traverse room_ptrs and check for match
.match_lp
    phx
    lda !animated_tiles,x
    and #$00ff
    cmp $1f5b
    bne .next
    lda !animated_tiles+15,x
    tax
.rooms_lp
    lda $df0000,x
    beq .next
    cmp $79b                ; room ID
    beq .match
    txa
    clc
    adc #$0006
    tax
    bra .rooms_lp

.match
    tyx
    lda $8085b6,x           ; dw 0001, 0002, 0004, 0008, 0010, 0020, 0040, 0080
    ora !minimap_settings+6
    sta !minimap_settings+6
    
.next
    plx
    txa
    clc
    adc #$0011
    tax
    iny : iny
    cpy #$000c
    bne .match_lp
  
    ply
    plx
    plp
    rts

assert pc() <= $82bf03
