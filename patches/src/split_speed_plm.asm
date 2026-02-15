; split speedbooster plms

lorom

!sbooster_plm_addr = $9600
!bbooster_plm_addr = $9700
!idx_SparkBooster = #$0016
!idx_BlueBooster = #$0017
!collect_item = $859980 ; this is a reference to the collect item routine location in stats.asm, go directly to it instead of replicating some of it here.

!bank_84_free_space_start = $84F0E2
!bank_84_free_space_end = $84F28B

org !bank_84_free_space_start
dw $EE64, sprk_boost        ; PLM $F0E2 (SparkBooster)
dw $EE64, sprk_boost_orb    ; PLM $F0E6 (SparkBooster, chozo orb)
dw $EE8E, sprk_boost_sce    ; PLM $F0EA (SparkBooster, scenery shot block)

dw $EE64, blue_boost        ; PLM $F0EE (BlueBooster)
dw $EE64, blue_boost_orb    ; PLM $F0F2 (BlueBooster, chozo orb)
dw $EE8E, blue_boost_sce    ; PLM $F0F6 (BlueBooster, scenery shot block)

;;; Instruction list - PLM $F0E2 (SparkBooster)
sprk_boost:
    dw $8764, !sbooster_plm_addr           ; Load item PLM GFX
    db $00, $00, $00, $00, $00, $00, $00, $00
    dw $887C, .end                         ; Go to end if the room argument item is set
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
.animate:
    dw $E04F                               ; Draw item frame 0
    dw $E067                               ; Draw item frame 1
    dw $8724, .animate                     ; loop
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (Stored shinespark Sound)
    db $0c                           
    dw collect_sb, $0080                   ; Pick up equipment $0080 and display message box $1f
    db $1f
.end
    dw $8724, $DFA9                        ; Go to $DFA9


;;; Instruction list - PLM $F0E6 (SparkBooster, chozo orb)
sprk_boost_orb:
    dw $8764, !sbooster_plm_addr           ; Load item PLM GFX
    db $00, $00, $00, $00, $00, $00, $00, $00
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
    dw $8724, .animate                     ; loop
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (Stored Shinespark Sound)
    db $0c                            
    dw collect_sb, $0080                   ; Pick up equipment $0080 and display message box $1f
    db $1f
.end
    dw $0001, $A2B5
    dw $86BC                               ; Delete


;;; Instruction list - PLM $F0EA (SparkBooster, scenery shot block)
sprk_boost_sce:
    dw $8764, !sbooster_plm_addr           ; Load item PLM GFX
    db $00, $00, $00, $00, $00, $00, $00, $00
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
    dw MISCFX                              ; Queue item sound (Stored shinespark Sound)
    db $0c                            
    dw collect_sb, $0080                   ; Pick up equipment $0080 and display message box $1f
    db $1F
.end
    dw $8A2E, $E032                        ; Call $E032 (empty item shot block reconcealing)
    dw $8724, .start
    
;;; Instruction list - PLM $F0EE (Bluebooster)
blue_boost:
    dw $8764, !bbooster_plm_addr           ; Load item PLM GFX
    db $03, $03, $00, $00, $03, $03, $00, $00
    dw $887C, .end                         ; Go to end if the room argument item is set
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
.animate:
    dw $E04F                               ; Draw item frame 0
    dw $E067                               ; Draw item frame 1
    dw $8724, .animate                     ; loop
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (speedboosting Sound)
    db $03                           
    dw collect_bb, $0040                   ; Pick up equipment $0040 and display message box $20
    db $20
.end
    dw $8724, $DFA9                        ; Go to $DFA9


;;; Instruction list - PLM $F0F2 (Bluebooster, chozo orb)
blue_boost_orb:
    dw $8764, !bbooster_plm_addr           ; Load item PLM GFX
    db $03, $03, $00, $00, $03, $03, $00, $00
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
    dw $8724, .animate                     ; loop
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (speedboosting Sound)
    db $03                           
    dw collect_bb, $0040                   ; Pick up equipment $0040 and display message box $20
    db $20
.end
    dw $0001, $A2B5
    dw $86BC                               ; Delete


;;; Instruction list - PLM $F0F6 (Bluebooster, scenery shot block)
blue_boost_sce:
    dw $8764, !bbooster_plm_addr           ; Load item PLM GFX
    db $03, $03, $00, $00, $03, $03, $00, $00
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
    dw MISCFX                              ; Queue item sound (speedboosting Sound)
    db $03                            
    dw collect_bb, $0040                   ; Pick up equipment $0040 and display message box $20
    db $20
.end
    dw $8A2E, $E032                        ; Call $E032 (empty item shot block reconcealing)
    dw $8724, .start
    
collect_sb:
    lda !idx_SparkBooster
    jmp collect
collect_bb:
    lda !idx_BlueBooster
collect:
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