\ Model Choquet
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.0953607082986934 y[1] + 0.0075476155853915 y[2]
   + 2.3358224719024069e-04 y[3] + 0.017490319829426 y[4]
   + 0.0500316606790774 y[5] + 0.1105317325359101 y[6]
   + 0.3170509638292943 y[7] + 0.037064406799026 y[8]
   + 0.0346522331477004 y[9] + 0.0324070692503939 y[10]
   + 0.0132246525341893 y[11] + 0.2084396084517524 y[12]
   + 0.055602624769579 y[13] + 0.0124048465736208 y[14]
   + 0.00795797546875522 y[15]
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
