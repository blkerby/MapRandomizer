lorom

!bank_84_free_space_start = $84F600
!bank_84_free_space_end = $84F700
!bank_85_free_space_start = $8596B0
!bank_85_free_space_end = $859800

org !bank_84_free_space_start

; These PLM entries must be at an address matching what is in `patch.rs`, starting here at $84F600
dw $EE64, inst        ; PLM $F600 (wall-jump boots)
dw $EE64, inst_orb    ; PLM $F604 (wall-jump boots, chozo orb)
dw $EE8E, inst_sce    ; PLM $F608 (wall-jump boots, scenery shot block)

;;; Instruction list - PLM $F600 (wall-jump boots)
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
    dw $88F3, $0400                        ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $8724, $DFA9                        ; Go to $DFA9


;;; Instruction list - PLM $F604 (wall-jump boots, chozo orb)
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
    dw $88F3, $0400                        ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $0001, $A2B5
    dw $86BC                               ; Delete


;;; Instruction list - PLM $F608 (wall-jump boots, scenery shot block)
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
    dw $88F3, $0400                        ; Pick up equipment $0400 and display message box $1E
    db $1E
.end
    dw $8A2E, $E032                        ; Call $E032 (empty item shot block reconcealing)
    dw $8724, .start


; from itemsounds.asm:
MISCFX:
	LDA #$0002
	STA $05D7
	LDA $0000,y
	INY
	JSL $80914D
	RTS

warnpc !bank_84_free_space_end

;;; repoint message box table references:

org $858250
    jsr (message_box_table+2, x)

org $858254
    jsr (message_box_table, x)

org $8582F1
    lda message_box_table+4,x

org $8582F6
    lda message_box_table+10,x

org !bank_85_free_space_start

;;; (message box table, relocated from $85869B):
message_box_table:
    dw $8436, $8289, $877F ; 1: Energy tank
    dw $83C5, $825A, $87BF ; 2: Missile
    dw $83C5, $825A, $88BF ; 3: Super missile
    dw $83C5, $825A, $89BF ; 4: Power bomb
    dw $83C5, $825A, $8ABF ; 5: Grappling beam
    dw $83CC, $825A, $8BBF ; 6: X-ray scope
    dw $8436, $8289, $8CBF ; 7: Varia suit
    dw $8436, $8289, $8CFF ; 8: Spring ball
    dw $8436, $8289, $8D3F ; 9: Morphing ball
    dw $8436, $8289, $8D7F ; Ah: Screw attack
    dw $8436, $8289, $8DBF ; Bh: Hi-jump boots
    dw $8436, $8289, $8DFF ; Ch: Space jump
    dw $83CC, $825A, $8E3F ; Dh: Speed booster
    dw $8436, $8289, $8F3F ; Eh: Charge beam
    dw $8436, $8289, $8F7F ; Fh: Ice beam
    dw $8436, $8289, $8FBF ; 10h: Wave beam (TODO: change back to $8FBF)
    dw $8436, $8289, $8FFF ; 11h: Spazer
    dw $8436, $8289, $903F ; 12h: Plasma beam
    dw $83C5, $825A, $907F ; 13h: Bomb
    dw $8436, $8289, $917F ; 14h: Map data access completed
    dw $8436, $8289, $923F ; 15h: Energy recharge completed
    dw $8436, $8289, $92FF ; 16h: Missile reload completed
    dw $8441, $8289, $93BF ; 17h: Would you like to save?
    dw $8436, $8289, $94BF ; 18h: Save completed
    dw $8436, $8289, $94FF ; 19h: Reserve tank
    dw $8436, $8289, $953F ; 1Ah: Gravity suit
    dw $8436, $8289, $957F ; 1Bh: Terminator
    dw $8441, $8289, $93BF ; 1Ch: Would you like to save? (Used by gunship)
    dw $8436, $8289, $94BF ; 1Dh: Terminator. (Save completed, unused)
    dw $8436, $8289, msg   ; 1Eh: Wall-jump boots.
    dw $8436, $8289, msg_end ; 1Fh: Terminator.

msg:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $2C0F, $2C0F, $2CD6, $2CC0, $2CCB, $2CCB, $2CDD, $2CC9, $2CD4, $2CCC, $2CCF, $2C0F, $2CC1, $2CCE, $2CCE, $2CD3, $2CD2, $2C0F, $2C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E

msg_end:

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
