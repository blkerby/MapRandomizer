lorom

incsrc "constants.asm"

!bank_84_free_space_start = $84F000
!bank_84_free_space_end = $84F100
!bank_85_free_space_start = $8596B0
!bank_85_free_space_end = $859800

!idx_WallJump = #$0015

org !bank_84_free_space_start

; These PLM entries must be at an address matching what is in `patch.rs`, starting here at $84F600
dw $EE64, inst        ; PLM $F000 (wall-jump boots)
dw $EE64, inst_orb    ; PLM $F004 (wall-jump boots, chozo orb)
dw $EE8E, inst_sce    ; PLM $F008 (wall-jump boots, scenery shot block)

;;; Instruction list - PLM $F000 (wall-jump boots)
inst:
    dw $8764, $9100                        ; Load item PLM GFX
    db $01, $01, $01, $01, $01, $01, $01, $01
    dw $887C, .end                         ; Go to end if the room argument item is set
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
.animate:
    dw $E04F                               ; Draw item frame 0
    dw $E067                               ; Draw item frame 1
    dw $8724, .animate                     ; Go to $E1FD
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (landing/walljump sound)
    db $04                            
    dw collect_WallJump, $0400             ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $8724, $DFA9                        ; Go to $DFA9


;;; Instruction list - PLM $F004 (wall-jump boots, chozo orb)
inst_orb:
    dw $8764, $9100                        ; Load item PLM GFX
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
    dw $8724, .animate                     ; Go to $E1FD
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (landing/walljump sound)
    db $04                            
    dw collect_WallJump, $0400                        ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $0001, $A2B5
    dw $86BC                               ; Delete


;;; Instruction list - PLM $F008 (wall-jump boots, scenery shot block)
inst_sce:
    dw $8764, $9100                        ; Load item PLM GFX
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
    db $04                            
    dw collect_WallJump, $0400                        ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $8A2E, $E032                        ; Call $E032 (empty item shot block reconcealing)
    dw $8724, .start


MISCFX:
    ; check if itemsounds.asm is applied
    LDA $848BF2
    CMP #$00A9
    BNE MISCFX_itemsounds
    ; fallthrough
    
NORMAL_QUEUE_MUSIC_ROUTINE:
    INY
    PHY
        ; points to byte with value 2 (item fanfare)
        ldy #MISCFX_itemsounds+1
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

; from stats.asm:
collect_item:
    jsl collect_item_85
    rts

collect_WallJump:
    lda !idx_WallJump
    jsr collect_item
    jmp $88F3

warnpc !bank_84_free_space_end

;;; repoint message box table references:

org $858243
    jmp hook_item_id
    
org $8582f1
    jsr hook_tilemap
    
org !bank_85_free_space_start
hook_item_id:
    lda $1c1f       ; replaced code
    cmp #$001e      ; wall-jump boots?
    beq .custom
    jmp $8246       ; resume normal func
    
.custom
    ldx #(message_box_table-$869b)
    jmp $824f       ; do ptr calls

hook_tilemap:
    cpx #$00ae      ; wall jump?
    bne .orig
    ldx #(message_box_table+4-$869f)
    
.orig
    lda $869f,x     ; replaced code
    rts

;;; (message box table, relocated from $85869B):
message_box_table:
    dw $8436, $8289, msg        ; 1Eh: Wall-jump boots.
    dw $8436, $8289, msg_end    ; 1Fh: Terminator.

msg:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $2C0F, $2C0F, $2CD6, $2CC0, $2CCB, $2CCB, $2CDD, $2CC9, $2CD4, $2CCC, $2CCF, $2C0F, $2CC1, $2CCE, $2CCE, $2CD3, $2CD2, $2C0F, $2C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E

msg_end:

collect_item_85:
    phx
    asl
    asl
    tax

    ; check if we have already collected one of this type of item (do not overwrite the collection time in this case):
    lda !stat_item_collection_times, x
    bne .skip
    lda !stat_item_collection_times+2, x
    bne .skip

    ; record the collection time
    lda !stat_timer
    sta !stat_item_collection_times, x
    lda !stat_timer+2
    sta !stat_item_collection_times+2, x
.skip:
    plx
    rtl

warnpc !bank_85_free_space_end


org $909E4B
    lda $09A2
    and #$0400
    bne walljump_enemy_check
    beq walljump_skip
warnpc $909E57
org $909E57
walljump_enemy_check:

org $909E6C
    lda $09A2
    and #$0400
    bne walljump_block_check
    beq walljump_skip
warnpc $909E78
org $909E78
walljump_block_check:
org $909E88
walljump_skip:
