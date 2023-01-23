\ Model Choquet
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.1263685270982235 y[0] + 0.104303435996953 y[1] + 0.0911065011174549 y[2]
   + 0.0227007379641371 y[3] + 0.00388886406891761 y[4]
   + 0.0406393630253274 y[5] + 0.1664082622560088 y[6]
   + 0.00122111778377669 y[7] + 0.0314992338703618 y[8]
   + 0.0203526676713187 y[9] + 0.00830072407981462 y[10]
   + 0.1211424253221833 y[11] + 0.0574213192150464 y[12]
   + 0.0632382426483326 y[13] + 0.1115937483444155 y[14]
   + 0.0298148295377281 y[15]
Subject To
 budget: 40 x[0] + 50 x[1] + 60 x[2] + 50 x[3] <= 100
 aptitude_0_0: - y[0] >= 0
 aptitude_1_0: - y[0] >= 0
 aptitude_0_1: 19 x[0] - y[1] >= 0
 aptitude_1_1: 2 x[0] - y[1] >= 0
 aptitude_0_2: 6 x[1] - y[2] >= 0
 aptitude_1_2: 11 x[1] - y[2] >= 0
 aptitude_0_3: 17 x[2] - y[3] >= 0
 aptitude_1_3: 4 x[2] - y[3] >= 0
 aptitude_0_4: 2 x[3] - y[4] >= 0
 aptitude_1_4: 18 x[3] - y[4] >= 0
 aptitude_0_5: 19 x[0] + 6 x[1] - y[5] >= 0
 aptitude_1_5: 2 x[0] + 11 x[1] - y[5] >= 0
 aptitude_0_6: 19 x[0] + 17 x[2] - y[6] >= 0
 aptitude_1_6: 2 x[0] + 4 x[2] - y[6] >= 0
 aptitude_0_7: 19 x[0] + 2 x[3] - y[7] >= 0
 aptitude_1_7: 2 x[0] + 18 x[3] - y[7] >= 0
 aptitude_0_8: 6 x[1] + 17 x[2] - y[8] >= 0
 aptitude_1_8: 11 x[1] + 4 x[2] - y[8] >= 0
 aptitude_0_9: 6 x[1] + 2 x[3] - y[9] >= 0
 aptitude_1_9: 11 x[1] + 18 x[3] - y[9] >= 0
 aptitude_0_10: 17 x[2] + 2 x[3] - y[10] >= 0
 aptitude_1_10: 4 x[2] + 18 x[3] - y[10] >= 0
 aptitude_0_11: 19 x[0] + 6 x[1] + 17 x[2] - y[11] >= 0
 aptitude_1_11: 2 x[0] + 11 x[1] + 4 x[2] - y[11] >= 0
 aptitude_0_12: 19 x[0] + 6 x[1] + 2 x[3] - y[12] >= 0
 aptitude_1_12: 2 x[0] + 11 x[1] + 18 x[3] - y[12] >= 0
 aptitude_0_13: 19 x[0] + 17 x[2] + 2 x[3] - y[13] >= 0
 aptitude_1_13: 2 x[0] + 4 x[2] + 18 x[3] - y[13] >= 0
 aptitude_0_14: 6 x[1] + 17 x[2] + 2 x[3] - y[14] >= 0
 aptitude_1_14: 11 x[1] + 4 x[2] + 18 x[3] - y[14] >= 0
 aptitude_0_15: 19 x[0] + 6 x[1] + 17 x[2] + 2 x[3] - y[15] >= 0
 aptitude_1_15: 2 x[0] + 11 x[1] + 4 x[2] + 18 x[3] - y[15] >= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3]
End
