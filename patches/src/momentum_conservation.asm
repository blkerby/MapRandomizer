; Patch to conserve Samus' horizontal momentum when landing while running
; Author: Scyzer, with updates by Nodever2 & OmegaDragnet7
lorom

!bank_90_free_space_start = $90F800
!bank_90_free_space_end = $90F880

lorom

; ASM to keep running speed while landing. This is more effective than simply changing the pose table,
; as you can have variables, and doesn't move you forward a pixel.

; By default, this patch only lets you speedkeep if you have speed booster equipped, are holding run,
; and are spin jumping. It is very easy to configure this patch to change all three of those things though.
; Please read the further comments for instructions.

;  Update 5-14-2022 by Nodever2 (rev1): Fix issue where if you use scyzer's suggested method of speedkeep with spin
;  jump, samus becomes much slipperier when turning around on the groung while running.
;  fixed by checking samus' last movement type instead and removing the BRA entirely.
;  I also changed the hijack point slightly, and fixed an issue with all speedkeep patches where
;  samus' palette is sometimes wrong for a couple frames when landing from screw attacking and speed boosting with speedkeep.

;  Update 6-26-2023 by OmegaDragnet7 (rev2): Fix a glitch where Samus becomes stuck midair when grappling a grappleable enemy.

org $90A3CA : JMP Landy ; hijack
SLOWDOWN: ; return point

org !bank_90_free_space_start
Landy:

; If you want to disable the check for the SpeedBooster item, comment the next line by putting a ; in front of the code.

;	LDA $09A2 : AND #$2000 : BEQ SLOW ; If speed booster not equipped, goto SLOW

; If you want to disable the check for RUN (so you don't need to be holding it to keep speed), comment the next line

	LDA $8B : AND $09B6 : BEQ SLOW ; If run not held, goto SLOW
	
; === Don't uncomment these lines ===	
;{
;   Grapple Fix by OmegaDragnet7
	LDA $0A1C : AND #$00AA : BEQ SLOW : AND #$00AB : BEQ SLOW 
	AND #$00B6 : BEQ SLOW : AND #$00B7 : BEQ SLOW : AND #$00A9 : BEQ SLOW
;   load samus' last different movement type
    LDA $0A27 : AND #$00FF
;}
; === End of required code section ===

; To keep speed even if you aren't spinning, uncomment the next line by removing the preceeding ;
	
	;CMP #$0002 : BEQ SPEEDKEEP : CMP #$0006 : BEQ SPEEDKEEP ; if last movement was normal jump or falling, speedkeep
	
	CMP #$0003 : BEQ SPEEDKEEP ; if last movement was spin jumping, speedkeep

SLOW: ; reset X speed
	JSR $9348 ; code that was replaced by hijack
	JMP SLOWDOWN
SPEEDKEEP:
	LDA $0B40 : BEQ + : LDA #$0003 : JSL $80914D ; resume speed booster sfx if needed
+	; next two lines of code are mostly only needed because of a dumb vanilla bug with $91DA74
	LDA #$0001 : STA $0AD0 ; update samus palette next frame
	LDA #$0004 : STA $0ACE ; reset samus speed booster/screw attack palette index
	PLP : RTS

warnpc !bank_90_free_space_end