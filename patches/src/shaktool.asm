org $84B8B0
    LDA #$0004
    CMP $0794   ; Check if we entered left vs right door
    BEQ left
    
    ; entered through right door
    LDA #$0100
    CMP $0AF6
    BCS set_shaktool  ; Go to set Shaktool flag if Samus X position <= $100
    RTS

left:
    ; entered through left door
    LDA #$0348 
    CMP $0AF6     ;} If Samus X position <= 348h, return without setting flag
    BCS done
set_shaktool:
    LDA #$000D   ;\
    JSL $8081FA  ;} Set Shaktool event
done:
    RTS

warnpc $84B8D6