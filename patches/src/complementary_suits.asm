;;;
;;; Complementary suits patch, by Maddo
;;;
;;; Based on Smiley's progressive suits patch.
;;;
;;; Effects :
;;;
;;; * Without Varia, Gravity provides no heat damage reduction and only 50% lava damage reduction.
;;; * Other suit combinations behave as in vanilla.

;;; compile with asar (https://www.smwcentral.net/?a=details&id=14560&p=section),
;;; or a variant of xkas that supports arch directive

lorom
arch snes.cpu

; heat check: $8DE379
; periodic damage: $90E9CE
; lava animation: $9081C0



; hook heat protection check
org $8DE379
	jsl check_heat_protection
	nop : nop
	bcs $2A

; hook heat damage calculation
org $8DE381
	jsl compute_heat_damage
	nop : nop : nop

;;; periodic damage modification (environmental damage)
org $90E9CE
env_damage:
	PHP
	REP #$30
	LDA $0A78        ;\
	BEQ .not_frozen  ;} If time is frozen: clear periodic damage and return
	JMP $EA3D        ;/
.not_frozen
	LDA $09A2
	BIT #$0020
	BEQ .no_gravity
	LDA $0A50        ;\ 
	LSR A            ;| Cut damage in half if Gravity is equipped
	STA $0A50        ;| (Vanilla ignores the upper byte of damage,
	LDA $0A4E        ;| but we have to handle it in order for
	ROR A            ;| Acid Chozo to function correctly.)
	STA $0A4E        ;| 
.no_gravity:
	LDA $09A2
	BIT #$0001   
	BEQ .no_varia
	LDA $0A4F     ;\
	LSR A         ;|  Cut damage in half if Varia is equipped
	STA $0A4F     ;/  (Ignore upper byte of damage and lower byte of subdamage.)
.no_varia:
	LDA $0A4C     ;\
	SEC           ;|
	SBC $0A4E     ;} Samus subhealth -= periodic subdamage
	STA $0A4C     ;/
	LDA $09C2     ;\
	SBC $0A50     ;} Samus health -= periodic damage
	STA $09C2     ;/
	STZ $0A4E     ; Periodic subdamage = 0
	STZ $0A50     ; Periodic damage = 0
	BPL .not_dead ; If Samus health < 0:
	STZ $0A4C     ; Samus subhealth = 0
	STZ $09C2     ; Samus health = 0
	PLP
	RTS
.not_dead
	PLP
	RTS

warnpc $90EA45

; Patch lava check code to skip damage only if both Varia and Gravity are equipped:
org $9081DB 
	jsr check_gravity_and_varia
	beq $2f

;;; free space in bank $90:
org $90F680
check_gravity_and_varia:
	and #$0021
	cmp #$0021
	rts

check_heat_protection:
	lda $09A2
	and #$0001       ; check if varia suit equipped
	bne .protected
	lda $0A44
	cmp #$E86A    	 ; check if samus is appearing (initial fanfare during new/loading game)
	beq .protected   
	clc
	rtl
.protected:
	sec
	rtl

compute_heat_damage:
	lda $09A2
	and #$0020       ; check if gravity suit equipped
	beq .no_gravity
	lda $0A4E
	clc
	adc #$8000       ; double heat damage if gravity suit is equipped, since it will get cut in half later
	rtl
.no_gravity:
	lda $0A4E
	clc
	adc #$4000       ; normal heat damage otherwise
	rtl

warnpc $90F700


;;; enemy damage division routine (suits patch)
;;; $12 is tmp var with enemy damage
org $a0a463
damage_div:
	bit #$0001		; A contains equipped items
	beq .novaria
	lsr $12			; /2 if varia
.novaria:
	bit #$0020
	beq .nogravity
	lsr $12			; /2 if grav
.nogravity:
	lda $12
	rtl

;;; metroid damage subroutine patch. (workaround hardcoded stuff??)
org $a3eed8
metroid_dmg:
	lda #$C000		; metroid damage value to dmg tmp var
	sta $12			;
	lda $09A2		; equipped items
	bit #$0020
	beq .nogravity
	lsr $12			; /2 if grav
.nogravity:
	bit #$0001
	beq .novaria		; /2 if varia
	lsr $12
.novaria:
	jmp $EEF2 		; continue routine

	