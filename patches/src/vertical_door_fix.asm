; Originally from https://forum.metroidconstruction.com/index.php/topic,145.msg73993.html#msg73993

; From PJ: (https://patrickjohnston.org/bank/82#fE4A9)
; Because scrolling updates take precedence over PLM draw updates, and because the scrolling updates were carried out prior to any PLM level data modifications,
; PLM draw updates that affect the top row of (the visible part of) the room for upwards doors or the bottom row of the room for downwards doors aren't visible
; This is the cause of the "red and green doors appear blue in the Crateria -> Red Brinstar room" bug

org $82E53C : JSL $808338 : JSL $8485B4 ; Waits for scrolling updates to happen before drawing the PLMs
