lorom

; Set the room Graphics flags bit $08 to use the Ceres tileset loading mode

;;; $E7D3: Load level data, CRE, tile table, scroll data, create PLMs and execute door ASM and room setup ASM ;;;
org $82E833 ; use the normal CRE/SCE loading routine
  LDA $07B3 ;LDA $079F
  BIT #$0008 ;CMP #$0006
  BNE $24 ;BEQ $24

;;; $EA73: Load level, scroll and CRE data ;;;
org $82EADB ; use the normal CRE/SCE loading routine
  LDA $07B3 ;LDA $079F
  BIT #$0008 ;CMP #$0006
  BNE $2C ;BEQ $2C

;;; $EA45: Pause check ;;;
org $90EA57 ; allow pausing in Ceres
  BRA + : NOP ;LDA $079F
  NOP : NOP : NOP ;CMP #$0006
  NOP : NOP ;BEQ $1E
+