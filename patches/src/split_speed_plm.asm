; split speedbooster plms

lorom

!bank_84_free_space_start = $84F4A0
!bank_84_free_space_end = $84F500

!idx_SparkBooster = #$0016
!idx_BlueBooster = #$0017

dw $EE64, sprk_boost        ; PLM $F4A0 (SparkBooster)
dw $EE64, sprk_boost_orb    ; PLM $F4A4 (SparkBooster, chozo orb)
dw $EE8E, sprk_boost_sce    ; PLM $F4A8 (SparkBooster, scenery shot block)

dw $EE64, blue_boost        ; PLM $F4AC (BlueBooster)
dw $EE64, blue_boost_orb    ; PLM $F4B0 (BlueBooster, chozo orb)
dw $EE8E, blue_boost_sce    ; PLM $F4B4 (BlueBooster, scenery shot block)

;;; Instruction list - PLM $F4A0 (SparkBooster)
sprk_boost:
    dw $8764, $9200                        ; Load item PLM GFX
    db $01, $01, $01, $01, $01, $01, $01, $01
    dw $887C, .end                         ; Go to end if the room argument item is set
    dw $8A24, .triggered                   ; Set link instruction for when triggered
    dw $86C1, $DF89                        ; Pre-instruction = go to link instruction if triggered
.animate:
    dw $E04F                               ; Draw item frame 0
    dw $E067                               ; Draw item frame 1
    dw $8724, .animate                     ; loop
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (ShineSpark Sound)
    db $0f                           
    dw collect, $0080             ; Pick up equipment $0080 and display message box $1f
    db $1f
.end
    dw $8724, $DFA9                        ; Go to $DFA9


;;; Instruction list - PLM $F4A4 (SparkBooster, chozo orb)
sprk_boost_orb:
    dw $8764, $9200                        ; Load item PLM GFX
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
    dw $8724, .animate                     ; loop
.triggered:
    dw $8899                               ; Set the room argument item
    dw MISCFX                              ; Queue item sound (ShineSpark Sound)
    db $0f                            
    dw collect, $0080                      ; Pick up equipment $0080 and display message box $1f
    db $1f
.end
    dw $0001, $A2B5
    dw $86BC                               ; Delete


;;; Instruction list - PLM $F4A8 (SparkBooster, scenery shot block)
sprk_boost_sce:
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
    dw MISCFX                              ; Queue item sound (ShineSpark Sound)
    db $0F                            
    dw colllect, $0080                     ; Pick up equipment $0080 and display message box $1f
    db $1F
.end
    dw $8A2E, $E032                        ; Call $E032 (empty item shot block reconcealing)
    dw $8724, .start
    
;;; Instruction list - PLM $F4AC (SparkBooster)
blue_boost:
    dw $8764, $9300                        ; Load item PLM GFX
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
    dw MISCFX                              ; Queue item sound (ShineSpark Sound)
    db $0f                           
    dw collect, $0080                      ; Pick up equipment $0080 and display message box $1f
    db $1f
.end
    dw $8724, $DFA9                        ; Go to $DFA9


;;; Instruction list - PLM $F4BO (SparkBooster, chozo orb)
blue_boost_orb:
    dw $8764, $9200                        ; Load item PLM GFX
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
    dw MISCFX                              ; Queue item sound (ShineSpark Sound)
    db $0f                            
    dw collect, $0080                      ; Pick up equipment $0080 and display message box $1f
    db $1f
.end
    dw $0001, $A2B5
    dw $86BC                               ; Delete


;;; Instruction list - PLM $F4B4 (SparkBooster, scenery shot block)
blue_boost_sce:
    dw $8764, $9100                        ; Load item PLM GFX
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
    dw MISCFX                              ; Queue item sound (ShineSpark Sound)
    db $0F                            
    dw colllect, $0080                     ; Pick up equipment $0080 and display message box $1f
    db $1F
.end
    dw $8A2E, $E032                        ; Call $E032 (empty item shot block reconcealing)
    dw $8724, .start