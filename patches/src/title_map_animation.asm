; Patch by BuggMann:

arch snes.cpu
LOROM


;This skips a palette rewrite before starting the animation
ORG $8BA359

 DW $A131

;This rewrites the animation instructions, greatly simplifies them, leaving $8BA13F to $8BA25A replaceable
ORG $8BA131

 DW $0006, $A337, $0006, $A341, $0006, $A34B, $94BC, $A131

;This realistically just rewrites the second value after the DWs, 
;but this could potentially let us animate other tiles down the road, so I kept it in for future use
;Basically it selects the beginning of the GFX file to animate those tiles
ORG $8BA337
 
 DB $C0
 DL $7F9000
 DW $0100, $0000
 DB $80, $00

 DB $C0
 DL $7F9100
 DW $0100, $0000
 DB $80, $00

 DB $C0
 DL $7F9200
 DW $0100, $0000
 DB $80, $00

;This imports the compressed mode 7 GFX data for the three frames of animation
ORG $95A5E1
    
    INCBIN mode7.bin