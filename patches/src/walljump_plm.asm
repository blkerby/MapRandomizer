; collectible walljump plm - this patch relies on extended_msg_boxes and custom_plm_gfx also being applied to function correctly.

lorom

!idx_WallJump = #$0015
!collect_item = $859980 ; this is a reference to the collect item routine location in stats.asm, go directly too it instead of replicating some of it here.

!bank_84_free_space_start = $84F000
!bank_84_free_space_end = $84F0E2

org !bank_84_free_space_start
dw $EE64, inst        ; PLM $F000 (wall-jump boots)
dw $EE64, inst_orb    ; PLM $F004 (wall-jump boots, chozo orb)
dw $EE8E, inst_sce    ; PLM $F008 (wall-jump boots, scenery shot block)

;;; Instruction list - PLM $F000 (wall-jump boots)
inst:
    dw $8764, $9100                        ; Load item PLM GFX (location of gfx is specified in custom_plm_gfx.asm)
    db $01, $01, $01, $01, $01, $01, $01, $01
    dw $887C, .end                         ; Go to end if the room argument item is set
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
.animate:
    dw $E04F                               ; Draw item frame 0
    dw $E067                               ; Draw item frame 1
    dw $8724, .animate                     ; 
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (landing/walljump sound)
    db $05                            
    dw collect_WallJump, $0400             ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $8724, $DFA9                        ; Go to $DFA9


;;; Instruction list - PLM $F004 (wall-jump boots, chozo orb)
inst_orb:
    dw $8764, $9100                        ; Load item PLM GFX (location of gfx is specified in custom_plm_gfx.asm)
    db $01, $01, $01, $01, $01, $01, $01, $01
    dw $887C, .end                         ; Go to end if the room argument item is set
    dw $8A2E, $DFAF                        ; Call $DFAF (item orb)
    dw $8A2E, $DFC7                        ; Call $DFC7 (item orb burst)
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
    dw $874E                               ; Timer = 16h
    db $16
.animate:
    dw $E04F                               ; Draw item frame 0
    dw $E067                               ; Draw item frame 1
    dw $8724, .animate                     ; 
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (landing/walljump sound)
    db $05                            
    dw collect_WallJump, $0400             ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $0001, $A2B5
    dw $86BC                               ; Delete


;;; Instruction list - PLM $F008 (wall-jump boots, scenery shot block)
inst_sce:
    dw $8764, $9100                        ; Load item PLM GFX (location of gfx is specified in custom_plm_gfx.asm)
    db $01, $01, $01, $01, $01, $01, $01, $01
.start:
    dw $8A2E, $E007                        ; Call $E007 (item shot block)
    dw $887C, .end                         ; Go to end if the room argument item is set
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
    dw $874E                               ; Timer = 16h
    db $16
.animate:
    dw $E04F                               ; Draw item frame 0
    dw $E067                               ; Draw item frame 1
    dw $873F, .animate
    dw $8A2E, $E020                        ; Call $E020 (item shot block reconcealing)
    dw $8724, .start                       ; Go to start
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (landing/walljump sound)
    db $05                            
    dw collect_WallJump, $0400             ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $8A2E, $E032                        ; Call $E032 (empty item shot block reconcealing)
    dw $8724, .start

collect_WallJump:
    lda !idx_WallJump
    jsl !collect_item
    jmp $88F3
    
MISCFX:
    ; check if itemsounds.asm is applied
    LDA $848BF2
    CMP #$00A9
    BNE MISCFX_itemsounds
    ; fallthrough
    
NORMAL_QUEUE_MUSIC_ROUTINE:
    INY
    PHY
        
    ldy #MISCFX_itemsounds+1 ; points to byte with value 2 (item fanfare)
    JSR $8BDD ; normal queue music routine
    PLY
    RTS

; from itemsounds.asm:
MISCFX_itemsounds:
	LDA #$0002 ; (punned; see above)
	STA $05D7
	LDA $0000,y
	INY
	JSL $80914D
	RTS

assert pc() <= !bank_84_free_space_end

org $909E4B
    lda $09A2
    and #$0400
    bne walljump_enemy_check
    beq walljump_skip
assert pc() <= $909E57

org $909E57
walljump_enemy_check:

org $909E6C
    lda $09A2
    and #$0400
    bne walljump_block_check
    beq walljump_skip
assert pc() <= $909E78

org $909E78
walljump_block_check:

org $909E88
walljump_skip: