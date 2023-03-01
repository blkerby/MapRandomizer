lorom

; Skip save confirmation message box:
org $848CF6
    nop : nop : nop : nop   ;$84:8CF6 22 80 80 85 JSL $858080[$85:8080]  ;} Display save confirmation message box
    nop : nop : nop         ;$84:8CFA C9 02 00    CMP #$0002             ;\
    nop : nop               ;$84:8CFD F0 33       BEQ $33    [$8D32]     ;} If [save confirmation selection] != no:

; Remove the game saved message box and return control immediately to Samus while
; sound and animation play.
org $84AFE8             
    dw $0001, $9A3F
    dw $86B4           ; Sleep
    dw $8CF1, $B008    ; Activate save station and go to $B008 if [save confirmation selection] = no
    dw $B00E           ; Place Samus on save station
    dw $8C07
    db $2E             ; Queue sound 2Eh, sound library 1, max queued sounds allowed = 6 (saving)
    dw $874E
    db $02             ; Replaces: db $15  (timer value)
    dw $0004, $9A9F
    dw $0004, $9A6F
    dw $873F, $AFFA    ; Decrement timer and go to $AFFA if non-zero
    dw $B024           ; Display game saved message box
    dw $B030           ; Enable movement and set save station used
    dw $8724, $AFE8    ; Go to $AFE8


; Reduce duration of save electricity (enemy projectile)
org $86E685
    dw $0001           ; normally $0014



;;; $AD86: Instruction list - PLM $B6D7 (collision reaction, special, BTS 47h. Map station right access) ;;;
org $84AD86             
    dw $8C10
    db $37    ; Queue sound 37h, sound library 2, max queued sounds allowed = 6 (refill/map station engaged)
    dw $0003, $9F49
    dw $0018, $9F55
    dw $8C8F       ; Activate map station
    dw $0003, $9F55
    dw $8C10
    db $38    ; Queue sound 38h, sound library 2, max queued sounds allowed = 6 (refill/map station disengaged)
    dw $0003, $9F55
    dw $0003, $9F49
    dw $86BC       ; Delete


;;; $ADA4: Instruction list - PLM $B6DB (collision reaction, special, BTS 48h. Map station left access) ;;;
org $84ADA4
    dw $8C10
    db $37    ; Queue sound 37h, sound library 2, max queued sounds allowed = 6 (refill/map station engaged)
    dw $0003, $9F5B
    dw $0018, $9F67
    dw $8C8F       ; Activate map station
    dw $0003, $9F67
    dw $8C10
    db $38    ; Queue sound 38h, sound library 2, max queued sounds allowed = 6 (refill/map station disengaged)
    dw $0003, $9F67
    dw $0003, $9F5B
    dw $86BC       ; Delete


;;; $ADF1: Instruction list - PLM $B6E3 (collision reaction, special, BTS 49h. Energy station right access) ;;;
; This is mostly unchanged from vanilla, only we change the frame counts to speed up the animation:
org $84ADF1
    dw $AE35, $AE11  ; Go to $AE11 and enable movement if Samus health is full
    dw $8C10
    db $37    ; Queue sound 37h, sound library 2, max queued sounds allowed = 6 (refill/map station engaged)
    dw $0003, $9FB5
    dw $0018, $9FBB
    dw $8CAF       ; Activate energy station
    dw $0003, $9FBB
    dw $8C10
    db $38    ; Queue sound 38h, sound library 2, max queued sounds allowed = 6 (refill/map station disengaged)
    dw $0003, $9FBB
    dw $0003, $9FB5
    dw $86BC        ; Delete


;;; $AE13: Instruction list - PLM $B6E7 (collision reaction, special, BTS 4Ah. Energy station left access) ;;;
; This is mostly unchanged from vanilla, only we change the frame counts to speed up the animation:
org $84AE13             
    dw $AE35, $AE33  ; Go to $AE33 and enable movement if Samus health is full
    dw $8C10 
    db $37   ; Queue sound 37h, sound library 2, max queued sounds allowed = 6 (refill/map station engaged)
    dw $0003, $9FC1
    dw $0018, $9FC7
    dw $8CAF       ; Activate energy station
    dw $0003, $9FC7
    dw $8C10
    db $38    ; Queue sound 38h, sound library 2, max queued sounds allowed = 6 (refill/map station disengaged)
    dw $0003, $9FC7
    dw $0003, $9FC1
    dw $86BC        ; Delete


;;; $AE7B: Instruction list - PLM $B6EF (collision reaction, special, BTS 4Bh. Missile station right access) ;;;
; This is mostly unchanged from vanilla, only we change the frame counts to speed up the animation:
org $84AE7B             
    dw $AEBF, $AE9B   ; Go to $AE9B and enable movement if full
    dw $8C10
    db $37    ; Queue sound 37h, sound library 2, max queued sounds allowed = 6 (refill/map station engaged)
    dw $0003, $9FB5
    dw $0018, $9FBB
    dw $8CD0       ; Activate missile station
    dw $0003, $9FBB
    dw $8C10
    db $38    ; Queue sound 38h, sound library 2, max queued sounds allowed = 6 (refill/map station disengaged)
    dw $0003, $9FBB
    dw $0003, $9FB5
    dw $86BC        ; Delete


;;; $AE9D: Instruction list - PLM $B6F3 (collision reaction, special, BTS 4Ch. Missile station left access) ;;;
; This is mostly unchanged from vanilla, only we change the frame counts to speed up the animation:
org $84AE9D             
    dw $AEBF, $AEBD  ; Go to $AEBD and enable movement if Samus missiles are full
    dw $8C10
    db $37    ; Queue sound 37h, sound library 2, max queued sounds allowed = 6 (refill/map station engaged)
    dw $0003, $9FC1
    dw $0018, $9FC7
    dw $8CD0       ; Activate missile station
    dw $0003, $9FC7
    dw $8C10
    db $38    ; Queue sound 38h, sound library 2, max queued sounds allowed = 6 (refill/map station disengaged)
    dw $0003, $9FC7
    dw $0003, $9FC1
    dw $86BC        ; Delete

;;;
; Saving at the gunship:
;;;

;;; $A5BE: Instruction list - gunship entrance pad - opening ;;;
; Speed up the animation: Cut the frames by about a factor of 4:
org $A2A5BE             
    dw $000A, $AFDD
    dw $0002, $AFC7
    dw $0002, $AE89
    dw $0002, $AE9F
    dw $0006, $AEB5
    dw $0002, $AECB
    dw $0002, $AEF5
    dw $0002, $AF1F
    dw $0002, $AF49
    dw $0001, $AF73
;org  $A2A5E6        
    dw $0001,$AF9D
    dw $80ED,$A5E6   ; Go to $A5E6


;;; $A5EE: Instruction list - gunship entrance pad - closing ;;;
; Speed up the animation: Cut the frames by about a factor of 4:
org $A2A5EE
    dw $0001, $AF73
    dw $0001, $AF49
    dw $0002, $AF1F
    dw $0002, $AEF5
    dw $0002, $AECB
    dw $0006, $AEB5
    dw $0002, $AE9F
    dw $0002, $AFC7


;;; $A60E: Instruction list - gunship entrance pad - closed ;;;
; Eliminate a small delay:
org $A2A60E             
    dw $0001, $AFDD
    dw $80ED, $A60E   ; Go to $A60E

; Eliminate the delay before Samus descends into the ship:
org $A2AA41 
    LDA #$0001         ; Replaces: LDA #$0090 

; Eliminate the delay after Samus descends into the ship:
org $A2AA86 
    LDA #$0001         ; Replaces: LDA #$0090 

; Eliminate the delay before Samus ascends from the ship:
org $A2AB52
    LDA #$0001         ; Replaces: LDA #$0090

; Eliminate the delay after Samus ascends from the ship:
org $A2AB97
    LDA #$0001         ; Replaces: LDA #$0090

; Remove the delay after starting the save sound effect:
org $858122
    LDA #$0001         ; Replaces: LDA #$00A0 

; Speed up refilling energy at the gunship:
org $A2AAAB 
    LDA #$0012         ; Replaces: LDA #$0002

; Speed up refilling Missiles at the gunship:
org $A2AAB2
    LDA #$0006         ; Replaces: LDA #$0002



;;;
; Changes that we decided not to make:
;;;

;org $8580BA
;done_ship_save:
;
;; Remove the "Save Completed" message when saving at the gunship:
;org $8580BF
;    JSR $846D         ; Handle message box interaction
;    JSR $8589         ; Close message box
;    LDA $05F9         ;\
;    CMP #$0002        ;} If [save confirmation selection] = yes:
;    BEQ skip_ship_save           ;/
;    LDA #$0018        ;\
;    STA $1C1F         ;} Message box index = save completed
;    JSR $81F3         ; Clear message box BG3 tilemap
;    JSR $8119         ; Play saving sound effect
;;    JSR $8241         ; Initialise message box
;;    JSR $8574         ; Play 2 lag frames of music and sound effects
;;    JSR $844C         ; Open message box
;;    JSR $846D         ; Handle message box interaction
;;    JSR $8589         ; Close message box
;skip_ship_save:
;    JSR $81F3         ; Clear message box BG3 tilemap
;    JSR $861A         ; Restore PPU
;    JSL $82BE2F       ; Queue Samus movement sound effects
;    JSR $8574         ; Play 2 lag frames of music and sound effects
;    JSR $80FA         ; Maybe trigger pause screen or return save confirmation selection
;    BRA done_ship_save  ; Return


; Increase speed of message box opening
;org $858458
;    ADC #$0400         ; replaces: ADC #$0200

; Increase speed of message box closing
; Note: We don't apply this change, to avoid throwing off timings (e.g. Moat CWJ)
;org $858592
;    SBC #$0400         ; replaces: SBC #$0200

